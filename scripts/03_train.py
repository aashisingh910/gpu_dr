#!/usr/bin/env python3
"""Step 03 - staged training of RETFound Plus-LAFT-XAI on the real EyePACS cache.

Objective
---------
    L = L_ordinal
      + 0.5 * L_hierarchical    any-DR / referable / STDR / PDR heads
      + 0.2 * L_boundary        |y - yhat|^gamma on the continuous severity
      + 0.2 * L_contrastive     ordinal-distance-weighted supervised contrastive
      + 0.3 * L_lesion          A3 expert supervision
      + 0.1 * L_PA              pathology/anatomy probe consistency
      + w(t) * L_XAI            attribution-lesion Dice, 0 -> 0.02 -> 0.05
      + 0.3 * L_CBF             class-balanced focal, a moderate auxiliary

Schedule
--------
    Stage 1  backbone frozen                       head 1e-4
    Stage 2  + GLA-LoRA adapters                   head 1e-4, lora 5e-5
    Stage 3  + last 33% of transformer blocks      head 5e-5, lora 5e-5, bb 5e-6
    Stage 4  near-full fine-tuning                 head 2e-5, lora 2e-5, bb 5e-6

After every epoch the per-sample losses are folded into the hard-example miner
and the sampler is rebuilt, so the next epoch draws harder on the Mild/Moderate/
Severe boundary cases the model is currently getting wrong.

Model selection is on 0.5*QWK + 0.3*MacroF1 + 0.2*minority-recall, never on
accuracy: a model that predicts "No DR" for everything scores 0.73 accuracy on
EyePACS and is clinically useless.

    python scripts/03_train.py --samples-per-epoch 4000 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Cap BLAS/OpenMP threads per process BEFORE numpy/torch import (they read
# these once, at import time). Prevents oversubscription with num_workers
# DataLoader worker processes each otherwise trying to use every core for
# their own numpy/cv2 calls - harmless if data loading was never the
# bottleneck (it measurably wasn't for this run - see history.json), but
# free insurance against it becoming one on a different machine/config.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import CACHE_DIR, OUT_DIR, RAW_DIR, StageCfg, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.data.eyepacs import CachedEyePACS  # noqa: E402
from dr.metrics import diagnostic_metrics, ordinal_metrics  # noqa: E402
from dr.model import build_model  # noqa: E402
from dr.screening import screening_selection_score  # noqa: E402
from dr.sampling import (HardExampleState, make_sampler,  # noqa: E402
                         mining_report, update_hardness)
from dr.modules.a3_lesion_moe import lesion_supervision_loss  # noqa: E402
from dr.modules.a5_fusion_gnn import pathology_anatomy_consistency  # noqa: E402
from dr.modules.a6_ordinal import (boundary_loss, coral_decode,  # noqa: E402
                                   class_balanced_focal_loss, dqk_loss,
                                   hierarchical_loss, ordinal_contrastive_loss,
                                   ordinal_loss)
from dr.modules.a9_xai import (attribution_consistency_loss,  # noqa: E402
                               xai_weight_for_stage)
from dr.modules.a10_uncertainty import build_calibrator  # noqa: E402
from dr.modules.a11_domain_adapt import dafa_loss  # noqa: E402
from dr.modules.external_lesion_batches import build_idrid_ddr_sample  # noqa: E402

BATCH_KEYS = ("image", "crops", "anatomy", "quality_axes", "hardness")


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def make_amp(cfg, device):
    """Wire up TrainCfg.amp, which was previously defined but never consulted
    anywhere in the training loop - dead config, same bug pattern as
    BackboneCfg.retfound_fallback was.

    bf16 needs no loss scaling and is safe on CPU and on CUDA Ampere+; fp16
    (older CUDA) needs a GradScaler. MPS autocast is documented in this repo
    as flaky for some ops, so it stays opt-in via the same flag rather than
    silently auto-enabling - this function will not turn amp on for MPS even
    if cfg.train.amp is True, since the config's own comment says why not.
    """
    enabled = bool(cfg.train.amp) and device.type in ("cuda", "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "cpu":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    use_scaler = enabled and device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if cfg.train.amp and device.type == "mps":
        print("[amp] cfg.train.amp is set but device is MPS - staying off "
              "(known-flaky autocast ops on this backend, see TrainCfg.amp)")
    elif enabled:
        print(f"[amp] enabled: autocast dtype={dtype}, "
              f"GradScaler={'on' if use_scaler else 'off (not needed for bf16)'}")
    return enabled, dtype, scaler


class EMA:
    """Exponential moving average of the trainable parameters only (not the
    frozen backbone - no memory cost there).

    Under grad_accum=8 (effective batch 16, per run_50pct.sh) and a frozen
    backbone, this run's entire trainable budget is 14-15M parameters seeing
    very few effective optimizer steps (a few thousand across an entire
    multi-stage run at this data scale) - each one noisier, proportionally,
    than in a full end-to-end fine-tune with a large batch. Polyak/EMA
    averaging is a standard, well-established fix for exactly this: it costs
    one shadow copy of the SMALL trainable parameter set and typically gives
    a measurably more stable evaluation/selection checkpoint. It was entirely
    absent from this training loop before now.
    """

    def __init__(self, params: list[torch.Tensor], decay: float = 0.995):
        self.decay = decay
        self.params = params
        self.shadow = [p.detach().clone() for p in params]

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self) -> list[torch.Tensor]:
        """Copies EMA weights into `self.params` in place; returns the raw
        weights so they can be restored with `swap_out`."""
        backup = [p.detach().clone() for p in self.params]
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s)
        return backup

    @torch.no_grad()
    def swap_out(self, backup: list[torch.Tensor]):
        for b, p in zip(backup, self.params):
            p.data.copy_(b)


def model_inputs(b: dict) -> dict:
    return {k: b[k] for k in BATCH_KEYS if k in b}


class AuxLesionSource:
    """Cycles through real IDRiD/DDR annotated images for auxiliary training.

    Kept separate from the EyePACS sampler entirely: these images have no DR
    grade, so they never enter `make_sampler`/`HardExampleState`, and their
    loss only touches the lesion evidence maps and the attribution map, not
    the ordinal/hierarchical/boundary/contrastive heads.
    """

    def __init__(self, root: Path, cfg, device, seed: int = 1337):
        from dr.data.lesion_datasets import discover_all
        records, found = discover_all(root)
        if not records:
            raise SystemExit(
                f"--aux-lesion-masks was set but no annotated records were "
                f"found under {root}. Run scripts/11_download_lesions.py "
                f"first, or drop --aux-lesion-masks.")
        self.records = records
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(records))
        self.pos = 0
        print(f"[aux] {len(records)} real ophthalmologist-annotated images "
              f"from {list(found)} available for auxiliary co-training")

    def _next_record(self):
        if self.pos >= len(self.order):
            self.order = self.rng.permutation(len(self.records))
            self.pos = 0
        rec = self.records[self.order[self.pos]]
        self.pos += 1
        return rec

    def next_batch(self, batch_size: int, mask_size: int):
        """Concatenate `batch_size` single-image samples into one batch.

        Skips unreadable images / images with no annotated pixels (see
        `build_idrid_ddr_sample`) and keeps drawing until it has enough, up
        to a generous retry cap so a few corrupt files can't hang training.
        """
        batches, targets, valids = [], [], []
        tries = 0
        while len(batches) < batch_size and tries < batch_size * 8:
            tries += 1
            rec = self._next_record()
            sample = build_idrid_ddr_sample(rec, self.cfg, self.device, mask_size)
            if sample is None:
                continue
            b, t, v = sample
            batches.append(b); targets.append(t); valids.append(v)
        if not batches:
            return None
        keys = batches[0].keys()
        merged = {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}
        target_masks = torch.cat(targets, dim=0)              # (B,n,S,S)
        valid = torch.stack(valids, dim=0)                    # (B,n)
        return merged, target_masks, valid


def selection_score(m: dict, weights: dict, minority: tuple) -> float:
    """0.5*QWK + 0.3*MacroF1 + 0.2*minority recall - the early-stopping target."""
    rec = np.nanmean([m.get(f"recall_grade_{g}", np.nan) for g in minority])
    parts = {"qwk": m.get("quadratic_weighted_kappa", 0.0),
             "macro_f1": m.get("f1_macro", 0.0),
             "minority_recall": 0.0 if not np.isfinite(rec) else rec}
    return float(sum(weights.get(k, 0.0) * v for k, v in parts.items()))


@torch.no_grad()
def evaluate_split(model, loader, device, n_grades: int = 5,
                   decode: str = "coral"):
    model.eval()
    P, Y, L, C = [], [], [], []
    for batch in loader:
        b = to_device(batch, device)
        out = model(model_inputs(b))
        P.append(out["ordinal"].class_probs.float().cpu().numpy())
        L.append(out["ordinal"].logits_ce.float().cpu().numpy())
        C.append(coral_decode(out["ordinal"].cum_probs).cpu().numpy())
        Y.append(b["grade"].cpu().numpy())
    probs = np.concatenate(P); y = np.concatenate(Y); logits = np.concatenate(L)
    # argmax over the differenced categorical cannot reach the interior grades
    # when the threshold gaps are narrow; rank decoding can.  See coral_decode.
    pred = np.concatenate(C) if decode == "coral" else probs.argmax(1)
    m = {**diagnostic_metrics(y, pred, probs), **ordinal_metrics(y, pred)}
    for g in range(n_grades):                 # per-grade recall drives selection
        mask = y == g
        m[f"recall_grade_{g}"] = float((pred[mask] == g).mean()) if mask.any() else float("nan")
    return m, probs, y, logits


def compute_losses(out: dict, b: dict, cfg, counts_t, w_xai: float, model=None):
    """The full objective; returns (total, per-term dict, per-sample loss).

    When cfg.loss.learned_weights is set, the seven hand-tuned LossCfg
    weights (w_ordinal..w_cbf) are replaced by `model.loss_weighting`
    (HomoscedasticLossWeighting): a learned log-variance per term instead of
    a fixed constant. lambda_XAI stays on its own stage schedule regardless -
    that gating is deliberate (see xai_weight_for_stage) and should not be
    re-learned away.
    """
    y = b["grade"]
    ordi = out["ordinal"]
    lo = cfg.loss

    l_ord = ordinal_loss(ordi, y)
    l_hier = hierarchical_loss(ordi, y, cfg.head.hierarchy_thresholds)
    l_bnd = boundary_loss(ordi, y, lo.boundary_gamma)
    l_con = ordinal_contrastive_loss(ordi.projection, y, lo.contrastive_temp)
    l_les = lesion_supervision_loss(
        out["moe"], b["lesion_masks"],
        (b["lesion_masks"].amax((2, 3)) > 0.5).float())
    l_pa = pathology_anatomy_consistency(out["fusion"])
    l_cbf = class_balanced_focal_loss(ordi.logits_ce, y, counts_t, lo.cb_beta,
                                      lo.focal_gamma,
                                      weight_power=cfg.sampling.class_weight_power)
    if w_xai > 0:
        l_xai = attribution_consistency_loss(out["attribution"], b["lesion_masks"])
    else:
        # keep the term in the log but out of the graph while lambda is zero
        l_xai = out["attribution"].sum().detach() * 0.0

    terms = {"ord": l_ord, "hier": l_hier, "bnd": l_bnd, "con": l_con,
             "les": l_les, "pa": l_pa, "cbf": l_cbf}

    # --- DR-NOVA: DQK-Loss, attacking the QWK-is-too-low weakness directly --
    l_dqk = None
    if getattr(lo, "use_dqk_loss", False):
        l_dqk = dqk_loss(ordi.class_probs, y, ordi.class_probs.shape[1])
        terms["dqk"] = l_dqk

    if model is not None and getattr(model, "loss_weighting", None) is not None:
        weighted_total, _ = model.loss_weighting(terms)
        total = weighted_total + w_xai * l_xai
    else:
        total = (lo.w_ordinal * l_ord + lo.w_hierarchical * l_hier
                 + lo.w_boundary * l_bnd + lo.w_contrastive * l_con
                 + lo.w_lesion * l_les + lo.w_pa * l_pa
                 + lo.w_cbf * l_cbf + w_xai * l_xai)
        if l_dqk is not None:
            total = total + lo.w_dqk * l_dqk

    # --- DR-NOVA: DAFA domain-adversarial term, attacking the domain-shift --
    # weakness. A genuine no-op whenever the batch carries no domain label -
    # exactly the case for ordinary single-corpus EyePACS training - so this
    # adds zero cost/effect until a labelled external-domain batch is mixed in.
    domain_logits = out.get("domain_logits")
    if domain_logits is not None and "domain_id" in b:
        l_dafa = dafa_loss(domain_logits, b["domain_id"])
        terms["dafa"] = l_dafa
        total = total + cfg.domain_adapt.loss_weight * l_dafa

    # per-sample loss for the hard-example miner: the sample-separable terms only
    with torch.no_grad():
        t = (y.unsqueeze(1) > torch.arange(ordi.cum_logits.shape[1],
                                           device=y.device).unsqueeze(0)).float()
        ps = F.binary_cross_entropy_with_logits(ordi.cum_logits, t,
                                                reduction="none").mean(1)
        ps = ps + lo.w_boundary * (ordi.expected_grade - y).abs() ** lo.boundary_gamma

    terms["xai"] = l_xai
    return total, terms, ps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--backbone", type=str, default=None)
    ap.add_argument("--retfound-ckpt", type=str, default=None,
                    help="path or 'repo:file' of real RETFound weights; "
                         "'none' clears it for a non-RETFound ablation")
    ap.add_argument("--allow-no-retfound", action="store_true",
                    help="ablation escape hatch; training normally hard-fails "
                         "when the RETFound checkpoint cannot be loaded")
    ap.add_argument("--pretrained", action="store_true",
                    help="initialise the backbone from timm ImageNet weights. "
                         "Only meaningful with --retfound-ckpt none, and the "
                         "only sane init for the ViT-Small baseline arm")
    ap.add_argument("--no-alpp", action="store_true",
                    help="disable the learned A2 preprocessing. ALPP sits in "
                         "front of the backbone, so even with every ViT block "
                         "frozen each step still backprops through all "
                         "1+n_crops passes to reach it - measured at ~30x the "
                         "step time. Off is what the pre-ALPP baseline arm ran")
    ap.add_argument("--lesion-encoder", type=Path, default=None,
                    help="checkpoint from scripts/10_lesion_pretrain.py")
    ap.add_argument("--stage-epochs", type=str, default=None,
                    help="comma-separated override, e.g. 2,3,3,3")
    ap.add_argument("--global-size", type=int, default=None)
    ap.add_argument("--crop-input", type=int, default=None)
    ap.add_argument("--n-crops", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--calib-batch", type=int, default=4,
                    help="batch for the GLA-LoRA calibration pass. Its gradient "
                         "term needs grads for the whole backbone at once, so on "
                         "a ViT-Large this is the peak-memory moment of the run")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--samples-per-epoch", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--tag", type=str, default="retfound_plus_laft_xai")
    ap.add_argument("--select", choices=("ordinal", "screening"), default=None,
                    help="model-selection objective. 'ordinal' = 0.5*QWK + "
                         "0.3*MacroF1 + 0.2*minority recall; 'screening' = the "
                         "fraction of patients cleared at the NPV constraint")
    ap.add_argument("--npv-target", type=float, default=None,
                    help="NPV the cleared bucket must hold under --select screening")
    ap.add_argument("--no-unfreeze", action="store_true",
                    help="force every stage to head+LoRA only. Backbone "
                         "unfreezing needs ~3.1 GB of accelerator memory for a "
                         "ViT-Large; use this when the run is memory-capped.")
    ap.add_argument("--calib-device", default=None,
                    help="device for the one-off GLA-LoRA importance probe. The "
                         "probe unfreezes the whole backbone and retains every "
                         "block output, so on a ViT-Large it peaks ~3.1 GB - "
                         "well above a memory-capped run's training footprint. "
                         "Pass 'cpu' to keep that spike off the accelerator; the "
                         "allocated ranks are identical either way.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute transformer block activations in the "
                         "backward pass instead of storing them. Costs ~30%% "
                         "throughput and buys back most of the activation "
                         "memory - the difference between fitting a ViT-Large "
                         "with six local crops under a hard memory cap and not.")
    ap.add_argument("--decode", choices=("coral", "argmax"), default="coral",
                    help="how to turn the ordinal head into a grade. 'coral' "
                         "counts exceeded thresholds (canonical); 'argmax' takes "
                         "the differenced categorical and cannot reach interior "
                         "grades when the threshold gaps are narrow.")
    ap.add_argument("--no-init-ladder-from-prior", action="store_true",
                    help="keep the CORAL ladder at its uniform data-independent "
                         "initialisation instead of seeding b_k = logit(P(Y>k)) "
                         "on the training class counts (Objective 1 fix, on by "
                         "default).")
    ap.add_argument("--no-learned-loss-weights", action="store_true",
                    help="use the fixed, hand-tuned LossCfg weights (w_ordinal, "
                         "w_hierarchical, ...) instead of learning one "
                         "log-variance per loss term (Kendall et al. 2018, on "
                         "by default). The learned version removes a whole "
                         "axis of manual tuning and rebalances automatically "
                         "as each term's natural scale shifts during training.")
    ap.add_argument("--amp", action="store_true",
                    help="enable mixed precision (bf16 on CPU/CUDA-Ampere+, "
                         "fp16+GradScaler on older CUDA). No effect on MPS - "
                         "known-flaky autocast ops there, stays off regardless.")
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="Polyak/EMA decay for the trainable parameters, "
                         "evaluated alongside the raw weights each epoch and "
                         "used for checkpoint selection when it scores better. "
                         "0 disables. Default (config) is 0.995, tuned for the "
                         "few-thousand-step regime a small dataset + grad_accum "
                         "produces - not the 0.999+ used in large-batch training.")
    ap.add_argument("--ladder-lr-mult", type=float, default=None,
                    help="train ordinal.b0/deltas at head_lr * this multiplier "
                         "instead of at head_lr directly. Default (config) is "
                         "10x; the evidence pack found the ladder moves by "
                         "~1e-4 over a full run at 1x, which is why Mild/"
                         "Severe recall is pinned at zero regardless of "
                         "sampling or loss re-weighting.")
    ap.add_argument("--xai-weight", type=str, default=None,
                    help="override the lambda_XAI stage schedule, e.g. '0.02' for a "
                         "constant weight in every stage or '0,0.02,0.05,0.05' for a "
                         "per-stage schedule. The default schedule holds lambda_XAI at "
                         "zero through stages 1-2 so attribution is not tied to lesion "
                         "masks while attention is still noise; a run that never reaches "
                         "stage 3 therefore never applies the attribution-consistency "
                         "loss at all, which is what this flag exists to fix.")
    ap.add_argument("--threads", type=int, default=None,
                    help="cap intra-op CPU threads (e.g. half the core count)")
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="stop cleanly once this much wall time has elapsed")
    ap.add_argument("--aux-lesion-masks", action="store_true",
                    help="Objective 4 follow-up: periodically fold a small "
                         "batch of real IDRiD/DDR ophthalmologist-annotated "
                         "images into training as an auxiliary lesion "
                         "(Dice+BCE) + attribution-consistency loss, using "
                         "`lesion_supervision_loss`'s `valid` mask so only "
                         "the four channels IDRiD/DDR actually annotate "
                         "(MA/HE/EX_H/EX_S) are scored. These images carry no "
                         "DR grade and never touch the ordinal/hierarchical/ "
                         "contrastive losses or the sampler - they only "
                         "sharpen the lesion evidence and attribution maps "
                         "the grading head and A9 explanations depend on.")
    ap.add_argument("--aux-lesion-root", type=Path, default=RAW_DIR / "lesion",
                    help="root of the downloaded IDRiD/DDR annotations")
    ap.add_argument("--aux-lesion-every", type=int, default=5,
                    help="fold in one auxiliary real-mask batch every N "
                         "training steps")
    ap.add_argument("--aux-lesion-batch", type=int, default=2,
                    help="images per auxiliary batch")
    ap.add_argument("--aux-lesion-weight", type=float, default=0.3,
                    help="weight applied to the auxiliary lesion+xai loss "
                         "before it is added to the main step's gradient")
    args = ap.parse_args()

    cfg = default_config()
    if args.batch_size: cfg.train.batch_size = args.batch_size
    if args.backbone: cfg.backbone.name = args.backbone
    if args.retfound_ckpt:
        cfg.backbone.retfound_ckpt = (
            None if args.retfound_ckpt.lower() == "none" else args.retfound_ckpt)
    if args.allow_no_retfound: cfg.backbone.require_retfound = False
    if args.pretrained: cfg.backbone.pretrained = True
    if args.no_alpp: cfg.preproc.alpp_enabled = False
    if args.global_size: cfg.preproc.global_size = args.global_size
    if args.crop_input: cfg.preproc.crop_input = args.crop_input
    if args.n_crops is not None: cfg.preproc.n_crops = args.n_crops
    if args.workers is not None: cfg.train.num_workers = args.workers
    if args.grad_accum: cfg.train.grad_accum = args.grad_accum
    if args.stage_epochs:
        eps = [int(x) for x in args.stage_epochs.split(",")]
        cfg.train.stages = tuple(
            StageCfg(s.name, e, s.train_backbone, s.train_lora, s.unfreeze_frac,
                     s.head_lr, s.lora_lr, s.backbone_lr, s.train_alpp)
            for s, e in zip(cfg.train.stages, eps))
    if args.select: cfg.train.select_objective = args.select
    if args.no_init_ladder_from_prior: cfg.train.init_ladder_from_prior = False
    if args.no_learned_loss_weights: cfg.loss.learned_weights = False
    if args.amp: cfg.train.amp = True
    if args.ema_decay is not None: cfg.train.ema_decay = args.ema_decay
    if args.ladder_lr_mult is not None: cfg.train.ladder_lr_mult = args.ladder_lr_mult
    if args.xai_weight:
        vals = tuple(float(x) for x in args.xai_weight.split(","))
        n = len(cfg.train.stages)
        cfg.xai.weight_schedule = vals if len(vals) > 1 else vals * n
        print(f"[setup] lambda_XAI schedule overridden -> {cfg.xai.weight_schedule}")
    if args.npv_target: cfg.screening.npv_target = args.npv_target
    if args.no_unfreeze:
        cfg.train.stages = tuple(
            StageCfg(s.name, s.epochs, False, s.train_lora, 0.0,
                     s.head_lr, s.lora_lr, 0.0, s.train_alpp)
            for s in cfg.train.stages)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg.device = args.device
    device = cfg.torch_device()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    print(f"[setup] device = {device}  threads = {torch.get_num_threads()}")
    if cfg.train.select_objective == "screening":
        print(f"[setup] selecting on SCREENING: maximise the fraction cleared "
              f"subject to NPV >= {cfg.screening.npv_target} for grade >= "
              f"{cfg.screening.task_grade} (sight-threatening DR)")
    if args.no_unfreeze:
        print("[setup] --no-unfreeze: every stage runs head+LoRA only")
    train_ds = CachedEyePACS(args.cache, "train", augment=True, cfg=cfg)
    val_ds = CachedEyePACS(args.cache, "val", augment=False, cfg=cfg)
    rng = np.random.default_rng(cfg.train.seed)
    if args.max_train and args.max_train < len(train_ds):
        train_ds.idx = rng.choice(train_ds.idx, args.max_train, replace=False)
    if args.max_val and args.max_val < len(val_ds):
        # monitoring subset only; the reported test metrics use the full split
        val_ds.idx = rng.choice(val_ds.idx, args.max_val, replace=False)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")
    print(f"[data] cache {train_ds.images.shape[1]}px -> global "
          f"{cfg.preproc.global_size}px + {cfg.preproc.n_crops} local crops "
          f"of {cfg.preproc.crop_size}px fed at {cfg.preproc.crop_input}px")
    if train_ds.crop_centres is None:
        print("[data] WARNING: no crops.npy in the cache; the local branch is "
              "falling back to a fixed tiling. Re-run 02_preprocess.py.")
    counts = train_ds.class_counts()
    print(f"[data] train grade counts = {counts.astype(int).tolist()}")

    labels = train_ds.labels()
    meta_idx = train_ds.meta_indices()
    # sized to the FULL cache, not the train-split count: `cache_row` (what
    # meta_idx/update_hardness/sampling_weights index by) is a global row
    # index assigned before the train/val/test split, so it ranges over the
    # whole cache, not 0..len(train subset)-1 - CachedEyePACS's own internal
    # `_hardness` buffer is already sized this way for the same reason.
    hard = HardExampleState.empty(len(train_ds.meta_full))

    def build_train_loader():
        sampler = make_sampler(labels, meta_idx, hard, cfg.sampling,
                               args.samples_per_epoch)
        return DataLoader(train_ds, batch_size=cfg.train.batch_size,
                          sampler=sampler, num_workers=cfg.train.num_workers,
                          drop_last=True, pin_memory=(device.type == "cuda"),
                          prefetch_factor=(4 if cfg.train.num_workers > 0 else None),
                          persistent_workers=cfg.train.num_workers > 0)

    val_ld = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers,
                        pin_memory=(device.type == "cuda"))

    # ---- build model ----------------------------------------------------
    model = build_model(cfg).to(device)
    print(f"[A4] backbone: {cfg.backbone.name}")

    # ---- resume support ---------------------------------------------------
    # This run may be killed at any point (unattended machine, no guarantee it
    # stays on) - `checkpoint.pt` is written after every completed epoch (see
    # below) and, if present here, means a previous invocation of this exact
    # command got partway through. Everything the rest of setup would
    # otherwise compute (LoRA adapter injection via calibration, the
    # lesion-encoder transfer, the CORAL ladder prior-init) is instead
    # reconstructed deterministically from the checkpoint so model.state_dict()
    # loads onto an architecturally-identical model, and training picks up at
    # the next epoch instead of restarting stage 1 from scratch.
    ckpt_path = out_dir / "checkpoint.pt"
    resume = None
    if ckpt_path.exists():
        print(f"[resume] found {ckpt_path} - continuing the run instead of "
              "starting over")
        resume = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        from dr.modules.a4_backbone_lora import inject_lora
        inject_lora(model.backbone.vit, resume["gla_lora_ranks"], cfg.backbone)
        model.adapters_ready = True
        missing, unexpected = model.load_state_dict(resume["model"], strict=False)
        if missing or unexpected:
            print(f"[resume] WARNING: {len(missing)} missing / {len(unexpected)} "
                  "unexpected keys loading the checkpointed model - continuing "
                  "anyway, but double-check --backbone/--n-crops/etc. match the "
                  "original command")
        model.to(device)
        info = resume["gla_lora_info"]
        print(f"[resume] LoRA adapters reinjected from saved ranks "
              f"{resume['gla_lora_ranks']}; model weights restored")

    print(f"[A4] {model.backbone.status}")
    if not model.backbone.load_report.get("loaded"):
        print("[A4] WARNING: running WITHOUT real RETFound weights "
              f"({model.backbone.load_report.get('reason')})")

    if args.grad_checkpoint:
        model.backbone.vit.set_grad_checkpointing(True)
        print("[A4] gradient checkpointing ON for the backbone blocks")

    # ---- optional: transfer the lesion-pretrained MoE encoder -----------
    if resume is not None:
        print("[A3] resuming: lesion-encoder transfer (if any) is already "
              "reflected in the loaded checkpoint")
    elif args.lesion_encoder and Path(args.lesion_encoder).exists():
        sd = torch.load(args.lesion_encoder, map_location="cpu", weights_only=False)
        # The lesion experts are convolutional and backbone-independent, but the
        # MoE gate projects the *global* backbone feature, so its width follows
        # embed_dim (1024 for a ViT-Large, 384 for a ViT-Small).  strict=False
        # tolerates missing keys but not shape mismatches, so drop the entries
        # that cannot fit and report them rather than failing the whole transfer.
        tgt = model.moe.state_dict()
        src = {k: v for k, v in sd["moe"].items()
               if k in tgt and tgt[k].shape == v.shape}
        skipped = sorted(set(sd["moe"]) - set(src))
        missing, unexpected = model.moe.load_state_dict(src, strict=False)
        if skipped:
            print(f"[A3] {len(skipped)} tensor(s) not transferable to this "
                  f"backbone width, left at init: {skipped[:4]}"
                  f"{' ...' if len(skipped) > 4 else ''}")
        print(f"[A3] lesion encoder transferred from {args.lesion_encoder} "
              f"({len(src)} tensors; missing {len(missing)}, unexpected {len(unexpected)}; "
              f"pretrained on {sd.get('sources')}, "
              f"val Dice {sd.get('best_dice', float('nan')):.4f})")
    else:
        print("[A3] no lesion-pretrained encoder supplied; the experts start "
              "from scratch and are supervised by the morphological priors only")

    # ---- GLA-LoRA rank allocation ---------------------------------------
    if resume is not None:
        print(f"[A4] resuming: LoRA ranks already reinjected above, skipping "
              f"the calibration pass")
    else:
        calib_ld = DataLoader(train_ds, batch_size=min(args.calib_batch, len(train_ds)),
                              shuffle=True)
        calib_dev = torch.device(args.calib_device) if args.calib_device else device
        if calib_dev != device:
            model.to(calib_dev)
        calib = to_device(next(iter(calib_ld)), calib_dev)
        cimg = (calib["image"] - model.norm_mean) / model.norm_std
        print(f"[A4] calibrating GLA-LoRA importance on {cimg.shape[0]} images "
              f"at {cimg.shape[-1]}px (device {calib_dev})", flush=True)
        info = model.fit_adapters(cimg, calib["lesion_masks"])
        del calib, cimg
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        model.to(device)
        print(f"[A4] GLA-LoRA importance S_l:  {info['importance_S']}")
        print(f"[A4]   gradient  G_l:          {info['grad_G']}")
        print(f"[A4]   lesion    L_l:          {info['lesion_L']}")
        print(f"[A4]   attention A_l:          {info['attn_A']}")
        print(f"[A4] allocated ranks:          {info['ranks']}")
        print(f"[A4] {info['n_at_max_rank']} blocks pinned to r_max; "
              f"{info['lora_params']:,} LoRA parameters")
        (out_dir / "gla_lora.json").write_text(json.dumps(info, indent=2))
        # per-block table (S_l/G_l/L_l/A_l/rank all have one entry per
        # transformer block) - the generic list/dict flattener in
        # dr.csv_export isn't the right shape for this one, so build it
        # directly rather than dumping the raw lists into single cells.
        save_csv_alongside(
            [{"block": i, "importance_S": s, "grad_G": g, "lesion_L": l_,
              "attn_A": a, "rank": r}
             for i, (s, g, l_, a, r) in enumerate(zip(
                 info["importance_S"], info["grad_G"], info["lesion_L"],
                 info["attn_A"], info["ranks"]))],
            out_dir / "gla_lora.json")

    counts_t = torch.as_tensor(counts)

    aux_source = None
    if args.aux_lesion_masks:
        aux_source = AuxLesionSource(args.aux_lesion_root, cfg, device)

    # ---- Objective 1 fix: seed the CORAL ladder at the training prior ----
    if resume is not None:
        print("[A6] resuming: CORAL ladder already reflects the loaded "
              "checkpoint's model state")
    elif cfg.train.init_ladder_from_prior:
        before = model.ordinal.thresholds().detach().cpu().numpy()
        after = model.init_ordinal_ladder_from_prior(counts_t).cpu().numpy()
        print(f"[A6] CORAL ladder re-initialised from the training class prior "
              f"(Objective 1 fix)")
        print(f"[A6]   b_k before (uniform init): {np.round(before, 4).tolist()}")
        print(f"[A6]   b_k after  (prior-matched): {np.round(after, 4).tolist()}")
    else:
        print("[A6] --no-init-ladder-from-prior: CORAL thresholds left at "
              "uniform initialisation")
    print(f"[A6] ordinal ladder trains at head_lr * {cfg.train.ladder_lr_mult:g} "
          f"(1.0 reproduces the old, effectively-frozen behaviour)")

    if resume is not None:
        history = resume["history"]
        best, best_epoch, stale = resume["best"], resume["best_epoch"], resume["stale"]
        epoch_global = resume["epoch_global"]
        rh = resume["hard"]
        hard = HardExampleState(rh["hardness"], rh["seen"], rh["boundary"], rh["lowconf"])
        torch.set_rng_state(resume["torch_rng_state"])
        if torch.cuda.is_available() and resume.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state_all"])
        np.random.set_state(resume["numpy_rng_state"])
        resume_stage, resume_epoch_in_stage = resume["stage_idx"], resume["epoch_in_stage"]
        print(f"[resume] continuing after stage {resume_stage+1} epoch "
              f"{resume_epoch_in_stage+1} (global epoch {epoch_global}, "
              f"best score {best:.4f} at epoch {best_epoch})")
    else:
        history, best, best_epoch = [], -1.0, 0
        stale = 0
        epoch_global = 0
        resume_stage, resume_epoch_in_stage = -1, -1
    resume_state_applied = resume is None
    t_start = time.time()
    stop = False

    for si, stage in enumerate(cfg.train.stages):
        if stop:
            break
        if resume is not None and si < resume_stage:
            continue   # fully completed in a previous run of this command
        gi = model.set_stage(stage)
        pr = model.param_report()
        print(f"\n{'='*70}\n[stage {si+1}/{len(cfg.train.stages)}] {stage.name} "
              f"| {stage.epochs} epochs")
        print(f"  A2 ALPP {'trainable' if gi.get('alpp_trainable') else 'frozen (avoids a full-depth backward)'}")
        print(f"  unfrozen blocks {gi['backbone']['unfrozen_blocks']}"
              f"/{gi['backbone']['total_blocks']}, LoRA "
              f"{'on' if gi['backbone']['lora_trainable'] else 'off'}")
        print(f"  LRs: head {stage.head_lr:.1e} | ladder "
              f"{stage.head_lr * cfg.train.ladder_lr_mult:.1e} | "
              f"lora {stage.lora_lr:.1e} | backbone {stage.backbone_lr:.1e}")
        print(f"  trainable {pr['trainable_params']:,} / {pr['total_params']:,} "
              f"({pr['trainable_pct']}%)\n{'='*70}", flush=True)

        groups = model.optimizer_param_groups(stage, cfg.train.weight_decay,
                                              cfg.train.ladder_lr_mult)
        if not groups:
            print("  nothing trainable in this stage, skipping")
            continue
        opt = torch.optim.AdamW(groups)
        train_ld = build_train_loader()
        steps = max(1, len(train_ld) * stage.epochs // max(cfg.train.grad_accum, 1))
        warm = max(1, int(steps * cfg.train.warmup_frac))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else
            0.5 * (1 + np.cos(np.pi * (s - warm) / max(steps - warm, 1))))
        params = [p for g in groups for p in g["params"]]
        amp_enabled, amp_dtype, scaler = make_amp(cfg, device)
        ema = EMA(params, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None

        start_epoch = 0
        if resume is not None and si == resume_stage and not resume_state_applied:
            opt.load_state_dict(resume["optimizer"])
            sched.load_state_dict(resume["scheduler"])
            if ema is not None and resume.get("ema_shadow") is not None:
                ema.shadow = [s.to(device) for s in resume["ema_shadow"]]
            start_epoch = resume_epoch_in_stage + 1
            resume_state_applied = True
            print(f"[resume] optimizer/scheduler/EMA state restored for stage "
                  f"{si+1}; continuing at epoch {start_epoch+1}/{stage.epochs}")

        for ep in range(start_epoch, stage.epochs):
            progress = (ep + 1) / max(stage.epochs, 1)
            w_xai = xai_weight_for_stage(cfg.xai, si, progress)
            model.train()
            agg = {k: 0.0 for k in ("total", "ord", "hier", "bnd", "con",
                                    "les", "pa", "cbf", "xai", "aux_les", "aux_xai")}
            n_seen = 0
            ep_idx, ep_loss, ep_pred, ep_true, ep_margin = [], [], [], [], []
            t0 = time.time()
            opt.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_ld):
                b = to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_enabled):
                    out = model(model_inputs(b))
                    y = b["grade"]
                    total, terms, per_sample = compute_losses(
                        out, b, cfg, counts_t, w_xai, model)

                scaled = total / cfg.train.grad_accum
                if scaler.is_enabled():
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                aux_les_val = aux_xai_val = 0.0
                if aux_source is not None and step % args.aux_lesion_every == 0:
                    aux_batch = aux_source.next_batch(args.aux_lesion_batch,
                                                      cfg.moe.mask_size)
                    if aux_batch is not None:
                        ab, a_targets, a_valid = aux_batch
                        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                            enabled=amp_enabled):
                            aux_out = model(model_inputs(ab))
                            a_les = lesion_supervision_loss(
                                aux_out["moe"], a_targets,
                                (a_targets.amax((2, 3)) > 0.5).float(), valid=a_valid)
                            a_xai = attribution_consistency_loss(
                                aux_out["attribution"],
                                a_targets * a_valid.view(*a_valid.shape, 1, 1))
                            aux_total = args.aux_lesion_weight * (a_les + a_xai)
                        aux_scaled = aux_total / cfg.train.grad_accum
                        if scaler.is_enabled():
                            scaler.scale(aux_scaled).backward()
                        else:
                            aux_scaled.backward()
                        aux_les_val, aux_xai_val = float(a_les), float(a_xai)

                if (step + 1) % cfg.train.grad_accum == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                        opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update()

                bs = y.shape[0]; n_seen += bs
                # Every .item()/.cpu() call below is its own CPU<->GPU sync
                # point - the CPU stops and waits for everything queued on
                # the GPU so far. The original code made ~13 of these EVERY
                # step (one per loss term, plus one per per-sample tracking
                # array), which serialises what should overlap and was
                # measured producing exactly the "GPU bursts to 99% then
                # sits at 0%" pattern rather than genuine compute-bound
                # utilisation. Batching every scalar into one tensor and
                # every per-sample array into one tensor, then transferring
                # each with a single .cpu() call, cuts that to 2 syncs per
                # step - same values, same order, just fetched together.
                term_keys = list(terms.keys())
                scalar_stack = torch.stack(
                    [total.detach()] + [terms[k].detach() for k in term_keys])
                scalar_vals = scalar_stack.cpu().numpy()
                agg["total"] += float(scalar_vals[0]) * bs
                # terms carries "dqk" whenever cfg.loss.use_dqk_loss (True by
                # default) and "dafa" whenever a batch carries a domain label
                # - neither is in agg's fixed initial key set, so accumulate
                # with .get() rather than assuming every key was preseeded.
                for k, v in zip(term_keys, scalar_vals[1:]):
                    agg[k] = agg.get(k, 0.0) + float(v) * bs
                agg["aux_les"] += aux_les_val * bs
                agg["aux_xai"] += aux_xai_val * bs

                with torch.no_grad():
                    p = out["ordinal"].class_probs
                    top2 = p.topk(2, dim=1).values
                    margin_t = top2[:, 0] - top2[:, 1]
                    pred_t = p.argmax(1)
                    # per_sample/margin are float, cache_row/pred/y are long -
                    # stacking needs one common dtype; float32 represents
                    # every value here (grades 0-4, cache rows < 2^24)
                    # exactly, so casting to int64 back on the CPU side after
                    # the transfer recovers the originals bit-for-bit.
                    per_sample_stack = torch.stack([
                        b["cache_row"].to(per_sample.dtype), per_sample,
                        pred_t.to(per_sample.dtype), margin_t,
                        y.to(per_sample.dtype)])
                idx_np, loss_np, pred_np, margin_np, true_np = per_sample_stack.cpu().numpy()
                ep_idx.append(idx_np.astype(np.int64))
                ep_loss.append(loss_np)
                ep_pred.append(pred_np.astype(np.int64))
                ep_margin.append(margin_np)
                ep_true.append(true_np.astype(np.int64))

                if step % 25 == 0:
                    el = time.time() - t0
                    aux_str = (f" aux_les {agg['aux_les']/n_seen:.3f} "
                              f"aux_xai {agg['aux_xai']/n_seen:.3f}"
                              if aux_source is not None else "")
                    print(f"  s{si+1}e{ep+1} step {step}/{len(train_ld)} "
                          f"loss={agg['total']/n_seen:.4f} "
                          f"(ord {agg['ord']/n_seen:.3f} hier {agg['hier']/n_seen:.3f} "
                          f"bnd {agg['bnd']/n_seen:.3f} con {agg['con']/n_seen:.3f} "
                          f"les {agg['les']/n_seen:.3f} pa {agg['pa']/n_seen:.3f} "
                          f"xai {agg['xai']/n_seen:.3f}@{w_xai:.3f}{aux_str}) "
                          f"{n_seen/max(el,1e-6):.2f} img/s", flush=True)

            # ---- hard-example mining -------------------------------------
            update_hardness(hard, np.concatenate(ep_idx), np.concatenate(ep_loss),
                            np.concatenate(ep_pred), np.concatenate(ep_true),
                            np.concatenate(ep_margin), cfg.sampling)
            mr = mining_report(hard, labels, meta_idx)
            train_ld = build_train_loader()

            vm, vprob, vyy, _ = evaluate_split(model, val_ld, device,
                                               cfg.head.n_grades, args.decode)
            used_ema = False
            if ema is not None:
                # Evaluate the EMA-averaged weights too, and use whichever
                # is better for this epoch's selection score - the raw
                # weights are what continue training either way.
                backup = ema.swap_in()
                vm_ema, vprob_ema, vyy_ema, _ = evaluate_split(
                    model, val_ld, device, cfg.head.n_grades, args.decode)
                ema.swap_out(backup)
                # screening_selection_score returns a dict ({"score": ..., "cleared": ...,
                # "npv": ..., ...}), not a bare number - comparing the dicts
                # themselves crashes with TypeError; the "score" field is the
                # actual scalar selection criterion.
                scr_raw = screening_selection_score(vyy, vprob, cfg.screening)
                scr_ema = screening_selection_score(vyy_ema, vprob_ema, cfg.screening)
                if scr_ema["score"] >= scr_raw["score"]:
                    vm, vprob, vyy, used_ema = vm_ema, vprob_ema, vyy_ema, True
                    print(f"  [ema] using EMA weights for this epoch's "
                          f"selection ({scr_ema['score']:.4f} >= {scr_raw['score']:.4f} raw)")
            scr = screening_selection_score(vyy, vprob, cfg.screening)
            vm.update({f"screen_{k}": v for k, v in scr.items()})
            if cfg.train.select_objective == "screening":
                score = scr["score"]
            else:
                score = selection_score(vm, cfg.train.select_weights,
                                        cfg.train.minority_grades)
            epoch_global += 1
            rec = {"epoch": epoch_global, "stage": stage.name, "stage_epoch": ep + 1,
                   "lambda_xai": w_xai,
                   "train_loss": agg["total"] / max(n_seen, 1),
                   "epoch_sec": round(time.time() - t0, 1),
                   "selection_score": score,
                   **({"learned_loss_weights": model.loss_weighting.as_dict()}
                      if getattr(model, "loss_weighting", None) is not None else {}),
                   **{f"train_{k}": agg[k] / max(n_seen, 1) for k in agg if k != "total"},
                   **{f"val_{k}": v for k, v in vm.items()},
                   **{f"mine_{k}": v for k, v in mr.items()}}
            history.append(rec)
            history_path = out_dir / "history.json"
            history_path.write_text(json.dumps(history, indent=2))
            save_csv_alongside(history, history_path)

            rec_min = np.nanmean([vm[f"recall_grade_{g}"]
                                  for g in cfg.train.minority_grades])
            print(f"[{stage.name} e{ep+1}] loss={rec['train_loss']:.4f} "
                  f"| QWK {vm['quadratic_weighted_kappa']:.4f} "
                  f"F1 {vm['f1_macro']:.4f} minRec {rec_min:.4f} "
                  f"| score {score:.4f} | acc {vm['accuracy']:.4f} "
                  f"refAUC {vm['auc_referable_dr']:.4f} ({rec['epoch_sec']}s)")
            print(f"           screening: cleared {scr['cleared']*100:5.1f}% "
                  f"at NPV {scr['npv']:.4f} (target {cfg.screening.npv_target}, "
                  f"{'MET' if scr['constraint_met'] else 'NOT MET'}) "
                  f"| thr {scr['threshold']:.3f} missed {scr['missed']} "
                  f"| STDR AUC {scr['auc']:.4f}")
            print(f"           per-grade recall "
                  f"{[round(vm[f'recall_grade_{g}'], 3) for g in range(5)]} "
                  f"| miner p90 H {mr['p90_hardness']:.2f} "
                  f"boundary {mr['boundary_rate']:.2f}", flush=True)

            if np.isfinite(score) and score > best:
                best, best_epoch, stale = score, epoch_global, 0
                # if EMA weights are what earned this score, the checkpoint
                # must actually contain them - swap in only for the save
                save_backup = ema.swap_in() if (ema is not None and used_ema) else None
                torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                            "gla_lora": info, "epoch": epoch_global,
                            "stage": stage.name, "val": vm, "score": score,
                            "ema": used_ema},
                           out_dir / "best.pt")
                if save_backup is not None:
                    ema.swap_out(save_backup)
                print(f"           saved new best (score {best:.4f}"
                      f"{', EMA weights' if used_ema else ''})")
            else:
                stale += 1
                if stale >= cfg.train.patience:
                    print(f"           early stop: {stale} epochs without gain")
                    stop = True

            # ---- checkpoint: written after EVERY epoch (not just new-best
            # epochs) so an unattended machine that shuts down mid-run never
            # loses more than the epoch in progress - re-running the exact
            # same command picks up right after this point instead of
            # restarting stage 1. Written to a temp file + atomic rename so a
            # shutdown mid-write can't corrupt the one checkpoint a resume
            # depends on.
            ckpt_payload = {
                "stage_idx": si, "epoch_in_stage": ep, "epoch_global": epoch_global,
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "ema_shadow": ([s.detach().cpu() for s in ema.shadow]
                              if ema is not None else None),
                "gla_lora_ranks": info["ranks"], "gla_lora_info": info,
                "hard": {"hardness": hard.hardness, "seen": hard.seen,
                         "boundary": hard.boundary, "lowconf": hard.lowconf},
                "history": history, "best": best, "best_epoch": best_epoch,
                "stale": stale,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (torch.cuda.get_rng_state_all()
                                       if torch.cuda.is_available() else None),
                "numpy_rng_state": np.random.get_state(),
                "args": vars(args),
            }
            tmp_ckpt = out_dir / "checkpoint.pt.tmp"
            torch.save(ckpt_payload, tmp_ckpt)
            os.replace(tmp_ckpt, ckpt_path)
            print(f"           [checkpoint] stage {si+1} epoch {ep+1} saved "
                  f"-> {ckpt_path}")

            if stop:
                break

            if args.time_budget_min and (time.time() - t_start) / 60 > args.time_budget_min:
                print(f"\n[budget] {args.time_budget_min} min elapsed, stopping cleanly")
                stop = True
                break

    # ---- guardrail: Objective 4's exact failure mode, made impossible to ----
    # ---- miss a second time -------------------------------------------------
    # The Objective 4 evidence pack's root cause was mechanical and silent:
    # lambda_XAI was 0 in every epoch of every run because no run ever reached
    # stage 3, and nobody checked history.json for it until an audit found it
    # after the fact. This check makes that finding automatic and immediate
    # instead of something that has to be independently rediscovered.
    max_lambda_xai = max((h.get("lambda_xai", 0.0) for h in history), default=0.0)
    if max_lambda_xai <= 0.0:
        print("\n" + "!" * 70)
        print("[WARNING] lambda_XAI was 0.0 in EVERY epoch of this run.")
        print("          The attribution-consistency loss (L_XAI) was never")
        print("          applied - this is the exact failure mode Objective 4's")
        print("          evidence pack diagnosed. Attribution will measure at")
        print("          chance (AUROC ~0.50) and this is EXPECTED, not a bug")
        print("          in scripts/21_xai_evaluation.py.")
        print(f"          This run reached stage {si + 1}/{len(cfg.train.stages)} "
              f"('{stage.name}'). XaiCfg.weight_schedule turns lambda_XAI on at "
              "stage 3 - either let the run reach it, or override with "
              "--xai-weight to apply L_XAI from stage 1.")
        print("!" * 70)
    else:
        print(f"\n[A9] lambda_XAI reached a peak of {max_lambda_xai:.4f} during "
              f"this run - the attribution-consistency loss WAS applied.")

    # ---- A10: fit the calibrator on validation logits -------------------
    print("\n[A10] fitting calibrator on the validation split")
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _, vprobs, vy, vlogits = evaluate_split(model, val_ld, device,
                                            cfg.head.n_grades, args.decode)
    cal = build_calibrator(cfg.uncertainty.calibrator, cfg.head.n_grades)
    cal.fit(torch.as_tensor(vlogits), torch.as_tensor(vy))
    torch.save({"kind": cfg.uncertainty.calibrator,
                "state": cal.state_dict() if hasattr(cal, "state_dict") else None},
               out_dir / "calibrator.pt")
    if hasattr(cal, "temperature"):
        print(f"[A10] fitted temperature T = {cal.temperature:.4f}")

    np.save(out_dir / "hardness.npy", hard.hardness)
    print(f"\n[done] best selection score {best:.4f} at epoch {best_epoch} "
          f"| total {(time.time()-t_start)/60:.1f} min | artefacts in {out_dir}")


if __name__ == "__main__":
    main()
