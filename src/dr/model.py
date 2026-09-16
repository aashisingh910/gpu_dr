"""RETFound Plus-LAFT-XAI: the assembled architecture.

    1024x1024 fundus (cached retinal field)
        |
        +-- global resize ----------------.
        |                                  |
        +-- N local lesion crops ------.   |
                                       v   v
                              A2 ALPP (learned fusion weights)
                                       |
                                       v
                    A3 Lesion MoE  <--- adaptive gate (Z_G, Z_L, Q, H)
                    A4 real RETFound + GLA-LoRA (shared global/local weights)
                                       |
                                       v
                    A5 bidirectional Pathology<->Anatomy fusion + graph
                                       |
                                A8 clinical fusion
                                       |
                    .------------------+------------------.
                    v                  v                  v
            A6 hierarchical      A7 Temporal*      A9 lesion-grounded XAI
            ordinal + boundary                     A10 uncertainty
            + contrastive

* A7 is instantiated but only driven when longitudinal data is supplied.

The local branch is the part that makes the resolution change real: the global
view supplies context at 448 px, while `n_crops` windows cut at native cache
resolution carry the lesion detail that a single 224 px whole-image view
destroys.  Both go through the *same* RETFound weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .modules.a2_preprocess import ALPP
from .modules.a3_lesion_moe import LesionMoE, MoEOutput
from .modules.a4_backbone_lora import RETFoundPlusBackbone
from .modules.a5_fusion_gnn import PathologyAnatomyFusion
from .modules.a6_ordinal import OrdinalHead, OrdinalOutput
from .modules.a7_temporal import TemporalPrognosticEngine
from .modules.a8_multimodal import MultimodalFusion, CLINICAL_FIELDS
from .modules.a9_xai import differentiable_attribution
from .modules.a11_domain_adapt import DomainAdversarialHead, dafa_lambda_schedule
from .modules.loss_weighting import HomoscedasticLossWeighting

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RETFoundPlusLAFTXAI(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.alpp = ALPP(cfg.preproc) if cfg.preproc.alpp_enabled else None
        self.backbone = RETFoundPlusBackbone(cfg.backbone, cfg.preproc.global_size)
        d_img = self.backbone.embed_dim
        self.moe = LesionMoE(cfg.moe, global_dim=d_img)
        self.fusion = PathologyAnatomyFusion(
            cfg.fusion, global_dim=d_img, lesion_dim=cfg.moe.lesion_dim,
            expert_dim=cfg.moe.expert_dim, n_experts=len(cfg.moe.experts))
        d_f = cfg.fusion.fused_dim
        self.multimodal = MultimodalFusion(d_f, len(CLINICAL_FIELDS),
                                           dropout=cfg.head.dropout)
        self.ordinal = OrdinalHead(d_f, cfg.head.n_grades, cfg.head.dropout,
                                   cfg.head.hierarchy_thresholds, cfg.head.proj_dim)
        self.temporal = TemporalPrognosticEngine(d_f, cfg.head.horizons,
                                                 dropout=cfg.head.dropout)
        # --- DR-NOVA: Lesion-Gated Ordinal Projection (LGOP) ----------------
        # This project's own design (no direct literature ancestor, like ALPP
        # and MR-LMoE), attacking the explainability/grounding weakness
        # measured directly (attribution AUROC 0.4217, Dice 0.0086, and a
        # counterfactual that points the WRONG way - erasing cited lesions
        # RAISED predicted severity). Root cause hypothesis: moe.z_lesion only
        # reaches the ordinal head indirectly, filtered through the full
        # fusion/GNN stack, so the classifier's decision is not mechanically
        # guaranteed to depend on lesion evidence at all. This adds an
        # explicit, learned-gated residual straight from pooled lesion
        # evidence into the ordinal head's input, so a gradient computed for
        # the grading loss has a direct path back to the lesion experts, and
        # erasing real lesion evidence has a direct path to changing the
        # prediction (the property the counterfactual test actually checks).
        self.lesion_gate_proj = nn.Linear(cfg.moe.lesion_dim, d_f)
        self.lesion_gate_scalar = nn.Linear(cfg.moe.lesion_dim, 1)
        # --- DR-NOVA: DAFA domain-adversarial head (A11) --------------------
        self.dafa = (DomainAdversarialHead(d_f, cfg.domain_adapt.n_domains,
                                           cfg.domain_adapt.hidden, cfg.domain_adapt.dropout)
                    if cfg.domain_adapt.enabled else None)
        self.loss_weighting = (
            HomoscedasticLossWeighting(("ord", "hier", "bnd", "con", "les", "pa", "cbf", "dqk"))
            if cfg.loss.learned_weights else None)
        self.adapters_ready = False
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("norm_mean", mean, persistent=False)
        self.register_buffer("norm_std", std, persistent=False)

    # -- A4 adapter allocation ------------------------------------------
    def fit_adapters(self, images: torch.Tensor, lesion_masks: torch.Tensor) -> dict:
        info = self.backbone.fit_adapters(images, lesion_masks)
        self.adapters_ready = True
        return info

    def set_stage(self, stage) -> dict:
        """Apply one stage of the fine-tuning schedule; returns param groups."""
        bb = self.backbone.set_stage(stage.train_backbone, stage.train_lora,
                                     stage.unfreeze_frac)
        # ALPP is pre-backbone: training it forces the gradient through every
        # block.  Only worth paying once something inside the backbone is
        # learning too (see StageCfg.train_alpp).
        train_alpp = getattr(stage, "train_alpp", True)
        if self.alpp is not None:
            for p in self.alpp.parameters():
                p.requires_grad = bool(train_alpp)
        groups = self.backbone.param_groups()
        head_params = [p for n, p in self.named_parameters()
                       if not n.startswith("backbone.") and p.requires_grad]
        return {"backbone": bb, "alpp_trainable": bool(train_alpp and self.alpp is not None),
                "n_head_params": sum(p.numel() for p in head_params),
                "n_lora_params": sum(p.numel() for p in groups["lora"]),
                "n_backbone_params": sum(p.numel() for p in groups["backbone"])}

    def optimizer_param_groups(self, stage, weight_decay: float,
                               ladder_lr_mult: float = 1.0) -> list[dict]:
        """Four learning rates: ordinal ladder, head, LoRA, backbone.

        The CORAL ladder (`ordinal.b0`, `ordinal.deltas`) used to be lumped
        into the generic "head" group and trained at the same 1e-4-ish rate
        as everything else. Objective 1's evidence pack found that under that
        regime the ladder drifts by ~1e-4 over an entire run - three orders
        of magnitude too small to reshape the per-grade decision bands - which
        is why Mild/Severe recall pins at zero no matter how the sampler or
        loss weights are tuned. `ladder_lr_mult` (default 1.0 = old
        behaviour) lets the ladder move at a multiple of the head LR so it can
        actually separate from wherever it was initialised.
        """
        groups = self.backbone.param_groups()
        bb_ids = {id(p) for p in self.backbone.parameters()}
        ladder_params = [p for p in (self.ordinal.b0, self.ordinal.deltas)
                         if p.requires_grad]
        ladder_ids = {id(p) for p in ladder_params}
        head = [p for p in self.parameters()
               if p.requires_grad and id(p) not in bb_ids and id(p) not in ladder_ids]
        out = []
        if head:
            out.append({"params": head, "lr": stage.head_lr,
                        "weight_decay": weight_decay, "name": "head"})
        if ladder_params:
            out.append({"params": ladder_params,
                        "lr": stage.head_lr * ladder_lr_mult,
                        "weight_decay": 0.0, "name": "ordinal_ladder"})
        if groups["lora"]:
            out.append({"params": groups["lora"], "lr": stage.lora_lr,
                        "weight_decay": 0.0, "name": "lora"})
        if groups["backbone"]:
            out.append({"params": groups["backbone"], "lr": stage.backbone_lr,
                        "weight_decay": weight_decay, "name": "backbone"})
        return out

    def init_ordinal_ladder_from_prior(self, class_counts) -> torch.Tensor:
        """See OrdinalHead.init_from_prior - re-seeds the CORAL thresholds at
        the training class prior instead of a uniform, data-independent gap.
        """
        return self.ordinal.init_from_prior(class_counts)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_report(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_params": tot, "trainable_params": tr,
                "trainable_pct": round(100.0 * tr / max(tot, 1), 3)}

    # -- preprocessing -----------------------------------------------------
    def _prepare(self, x01: torch.Tensor, q_axes: torch.Tensor | None
                 ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the learned ALPP, then ImageNet-normalise for the backbone."""
        w = None
        if self.alpp is not None and "alpp" not in set(self.cfg.ablate):
            x01, w = self.alpp(x01, q_axes)
        return (x01 - self.norm_mean) / self.norm_std, w

    # -- forward ----------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        x01 = batch["image"]                       # (B,3,Hg,Wg) in [0,1]
        anat = batch["anatomy"]
        crops01 = batch.get("crops")               # (B,C,3,Hc,Wc) in [0,1]
        q_axes = batch.get("quality_axes")
        hardness = batch.get("hardness")
        B = x01.shape[0]
        ablate = set(self.cfg.ablate)
        if "local" in ablate:
            crops01 = None

        x, alpp_w = self._prepare(x01, q_axes)
        crops = None
        if crops01 is not None:
            Bc, C = crops01.shape[:2]
            cflat, _ = self._prepare(crops01.flatten(0, 1), None)
            crops = cflat.view(Bc, C, *cflat.shape[1:])

        bb = self.backbone(x, crops)

        # the MoE reads the same enhanced pixels the backbone does, at the
        # resolution its evidence maps are defined on
        moe_in = x if x.shape[-1] <= 512 else F.interpolate(
            x, size=(448, 448), mode="bilinear", align_corners=False)
        moe: MoEOutput = self.moe(moe_in, z_global=bb.cls.detach(),
                                  quality=q_axes, hardness=hardness)

        # --- novelty fix: also run the lesion MoE on each local crop -------
        # The global view above is a 448px resize of a 1024px field - exactly
        # the resolution collapse Objective 2's evidence pack shows destroys
        # microaneurysm contrast. The local crops are native-resolution
        # windows built to survive that collapse, but until now nothing
        # lesion-specific ever looked at them. Re-using the SAME moe weights
        # per crop and taking an elementwise max of presence logits (global
        # vs. best crop) means a lesion visible only at crop resolution is no
        # longer silently dropped by the global-only pass.
        if (crops is not None and bb.local_cls is not None
                and self.cfg.moe.use_crop_evidence and "crop_moe" not in ablate):
            Bc2, C2 = crops.shape[:2]
            crop_moe_in = crops.flatten(0, 1)
            if crop_moe_in.shape[-1] > 512:
                crop_moe_in = F.interpolate(crop_moe_in, size=(448, 448),
                                            mode="bilinear", align_corners=False)
            z_global_c = bb.local_cls.detach().flatten(0, 1)
            q_c = (q_axes.unsqueeze(1).expand(Bc2, C2, q_axes.shape[-1]).reshape(Bc2 * C2, -1)
                  if q_axes is not None else None)
            h_c = (hardness.view(Bc2, 1).unsqueeze(1).expand(Bc2, C2, 1).reshape(Bc2 * C2, 1)
                  if hardness is not None else None)
            moe_crop: MoEOutput = self.moe(crop_moe_in, z_global=z_global_c,
                                           quality=q_c, hardness=h_c)
            crop_presence_pooled = moe_crop.presence.view(Bc2, C2, -1).amax(1)  # (B, n_experts)
            moe = MoEOutput(z_lesion=moe.z_lesion, alphas=moe.alphas,
                            evidence=moe.evidence,
                            presence=torch.maximum(moe.presence, crop_presence_pooled),
                            expert_tokens=moe.expert_tokens)
        if "moe" in ablate:
            # drop the lesion pathway: experts contribute no signal to fusion
            moe = MoEOutput(
                z_lesion=torch.zeros_like(moe.z_lesion),
                alphas=moe.alphas,
                evidence=moe.evidence,
                presence=moe.presence,
                expert_tokens=torch.zeros_like(moe.expert_tokens),
            )
        fus = self.fusion(bb.tokens, bb.grid, moe.expert_tokens, moe.z_lesion,
                          anat, None if "fusion" in ablate else moe.evidence,
                          local_cls=bb.local_cls)

        clinical = batch.get("clinical")
        if clinical is None:
            clinical = torch.zeros(B, len(CLINICAL_FIELDS), device=x.device, dtype=x.dtype)
        valid = batch.get("clinical_valid")
        if valid is None or "clinical" in ablate:
            valid = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        mm = self.multimodal(fus.z_fused, clinical, valid)

        # --- DR-NOVA: Lesion-Gated Ordinal Projection ------------------------
        # z_mc + gate(z_lesion) * proj(z_lesion): an explicit residual path from
        # pooled lesion evidence into the classifier input, on top of whatever
        # already arrived indirectly through fusion. gate is per-sample and
        # learned, not fixed, so the model can still down-weight it for images
        # the lesion experts are uncertain about. When "moe" is ablated,
        # moe.z_lesion is already zeroed above, so this residual correctly
        # contributes nothing without any special-casing here.
        z_mc = mm.z_mc
        if "lesion_gate" not in ablate:
            lesion_gate = torch.sigmoid(self.lesion_gate_scalar(moe.z_lesion))
            z_mc = z_mc + lesion_gate * self.lesion_gate_proj(moe.z_lesion)

        ordi: OrdinalOutput = self.ordinal(z_mc)
        attribution = differentiable_attribution(
            fus.attn_g2l, self.fusion.pool_grid, self.cfg.moe.mask_size,
            n_spatial=self.fusion.pool_grid ** 2)

        # --- DR-NOVA: DAFA domain-adversarial head ---------------------------
        # Computed whenever enabled (cheap - one small MLP); only affects
        # gradients when the training loop supplies batch["domain_id"], via
        # the loss function that consumes "domain_logits" below. Gated by
        # A1's own Q_domain axis (quality_axes[:,1]) when available, so an
        # atypical/low-quality image - where the domain label itself is least
        # trustworthy - contributes less to the alignment gradient.
        domain_logits = None
        if self.dafa is not None and "domain_adapt" not in ablate:
            progress = float(batch.get("_train_progress", 1.0))
            lambd = dafa_lambda_schedule(progress, self.cfg.domain_adapt.grl_gamma)
            dom_weight = q_axes[:, 1] if q_axes is not None and q_axes.shape[-1] > 1 else None
            domain_logits = self.dafa(fus.z_fused, lambd, dom_weight)

        return {
            "moe": moe,
            "backbone_cls": bb.cls,
            "local_cls": bb.local_cls,
            "alpp_weights": alpp_w,
            "fusion": fus,
            "z_mc": z_mc,
            "clinical_gate": mm.gate,
            "ordinal": ordi,
            "attribution": attribution,
            "domain_logits": domain_logits,
            # clinical read-outs derived from the monotone cumulative head
            "referable": ordi.cum_probs[:, 1],          # P(grade >= 2)
            "sight_threatening": ordi.cum_probs[:, 2],  # P(grade >= 3)
        }

    @torch.no_grad()
    def embed_visit(self, batch: dict) -> torch.Tensor:
        """Per-visit representation consumed by the A7 temporal engine."""
        return self.forward(batch)["z_mc"]


def build_model(cfg: Config) -> RETFoundPlusLAFTXAI:
    torch.manual_seed(cfg.train.seed)
    return RETFoundPlusLAFTXAI(cfg)
