"""A4 - RETFound backbone + Gradient-Lesion Adaptive LoRA (GLA-LoRA).

Two things changed here.

1) The checkpoint is real, or training stops.
   `load_retfound_weights` loads genuine RETFound CFP MAE weights, reports the
   exact missing/unexpected key counts, and refuses to continue on a silent
   miss when `require_retfound` is set.  There is no ImageNet fallback: a run
   that quietly trains an `augreg_in21k` ViT and calls itself RETFound is worse
   than a run that fails.

2) The adapter rank is allocated from three signals, not one.
   The previous version scored a layer only by how well its token energy
   correlated with lesion density.  That is one noisy estimator.  GLA-LoRA
   pools three:

       S_l = lam_g * G_l  +  lam_l * L_l  +  lam_a * A_l

       G_l  gradient sensitivity - the norm of dL/dW inside block l on a
            calibration batch, i.e. how much the task actually wants to move
            this layer
       L_l  lesion relevance     - |corr(token energy, lesion density)|
       A_l  lesion attention mass - the share of each block's attention that
            lands on lesion-positive patches

   then

       r_l = r_min + (r_max - r_min) * S_l

   with layers above `lesion_layer_boost` pinned to r_max.  Low-importance
   layers get the floor rank, lesion-sensitive layers get the ceiling.

3) Staged fine-tuning.
   `set_stage` implements the frozen -> LoRA -> partial-unfreeze -> near-full
   schedule by flipping requires_grad on the last `unfreeze_frac` of blocks,
   and hands back the parameter groups so the optimizer can give the head, the
   adapters and the backbone their three different learning rates.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import BackboneCfg


# --------------------------------------------------------------------------
# LoRA
# --------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """y = W0 x + (alpha/r) * B(A(x)) with W0 frozen at injection time."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = int(rank)
        self.scale = alpha / max(self.rank, 1)
        self.drop = nn.Dropout(dropout)
        if self.rank > 0:
            self.A = nn.Parameter(torch.empty(self.rank, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, self.rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # B stays 0 -> identity at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.rank > 0:
            out = out + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)
        return out

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scale={self.scale:.3f}"


def _iter_blocks(model: nn.Module):
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("backbone has no .blocks - expected a timm ViT")
    return blocks


def inject_lora(model: nn.Module, ranks: list[int], cfg: BackboneCfg) -> int:
    """Replace the target nn.Linear layers of each block with LoRALinear.

    Returns the number of trainable LoRA parameters added.
    """
    added = 0
    for i, blk in enumerate(_iter_blocks(model)):
        r = ranks[i]
        for target in cfg.lora_targets:
            parent, name = None, None
            if target == "qkv" and hasattr(blk.attn, "qkv"):
                parent, name = blk.attn, "qkv"
            elif target == "proj" and hasattr(blk.attn, "proj"):
                parent, name = blk.attn, "proj"
            if parent is None:
                continue
            base = getattr(parent, name)
            if not isinstance(base, nn.Linear):
                continue
            lora = LoRALinear(base, r, cfg.lora_alpha, cfg.lora_dropout)
            setattr(parent, name, lora)
            if r > 0:
                added += lora.A.numel() + lora.B.numel()
    return added


def lora_parameters(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.rank > 0:
            yield m.A
            yield m.B


# --------------------------------------------------------------------------
# GLA-LoRA: the three importance signals
# --------------------------------------------------------------------------
def _patch_grid(n_tokens: int) -> int | None:
    g = int(math.isqrt(n_tokens))
    return g if g * g == n_tokens else None


@torch.no_grad()
def _lesion_relevance(backbone: nn.Module, images: torch.Tensor,
                      lesion_masks: torch.Tensor) -> torch.Tensor:
    """L_l - |corr(patch-token energy, lesion density)| per block."""
    feats: list[torch.Tensor] = []
    hooks = [blk.register_forward_hook(
        lambda _m, _i, o: feats.append(o.detach() if isinstance(o, torch.Tensor)
                                       else o[0].detach()))
        for blk in _iter_blocks(backbone)]
    was_training = backbone.training
    backbone.eval()
    backbone.forward_features(images)
    for h in hooks:
        h.remove()
    backbone.train(was_training)

    npx = getattr(backbone, "num_prefix_tokens", 1)
    dens_full = lesion_masks.amax(1, keepdim=True).float()          # (B,1,S,S)
    scores = []
    for f in feats:
        if f.dim() != 3:
            scores.append(torch.tensor(0.0, device=images.device))
            continue
        tok = f[:, npx:, :]
        g = _patch_grid(tok.shape[1])
        if g is None:
            scores.append(torch.tensor(0.0, device=images.device))
            continue
        energy = tok.float().pow(2).mean(-1).reshape(-1, 1, g, g)
        dens = F.adaptive_avg_pool2d(dens_full, (g, g))
        a = energy.flatten(1); b = dens.flatten(1)
        a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
        corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
        scores.append(corr.abs().mean())
    return torch.stack(scores)


@torch.no_grad()
def _lesion_attention_mass(backbone: nn.Module, images: torch.Tensor,
                           lesion_masks: torch.Tensor) -> torch.Tensor:
    """A_l - share of each block's attention that lands on lesion patches.

    Recomputes q@k inside a forward hook (timm fuses attention through SDPA, so
    the probabilities are not otherwise observable).  The score is the mean
    attention weight over lesion-positive patch keys divided by the mean over
    all patch keys, so a block that attends to lesions no more than chance
    scores 1.0 and is then re-centred to 0.
    """
    masses: list[torch.Tensor] = []
    npx = getattr(backbone, "num_prefix_tokens", 1)

    def make_hook(block):
        def hook(_m, inp, _out):
            x = inp[0]
            B, N, C = x.shape
            attn = block.attn
            qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads,
                                      C // attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            a = ((q @ k.transpose(-2, -1)) * attn.scale).softmax(-1)   # (B,H,N,N)
            a = a.mean(1)[:, :, npx:]                                  # keys = patches
            g = _patch_grid(a.shape[-1])
            if g is None:
                masses.append(torch.tensor(1.0, device=x.device))
                return
            dens = F.adaptive_avg_pool2d(lesion_masks.amax(1, keepdim=True).float(),
                                         (g, g)).flatten(1)            # (B, P)
            pos = (dens > 0.05).float()
            frac = pos.mean(1).clamp(1e-3, 1 - 1e-3)                   # (B,)
            on_lesion = (a.mean(1) * pos).sum(1)                       # (B,)
            masses.append((on_lesion / frac).mean())
        return hook

    hooks = [blk.register_forward_hook(make_hook(blk)) for blk in _iter_blocks(backbone)]
    was_training = backbone.training
    backbone.eval()
    backbone.forward_features(images)
    for h in hooks:
        h.remove()
    backbone.train(was_training)
    if not masses:
        return torch.zeros(len(list(_iter_blocks(backbone))), device=images.device)
    return (torch.stack(masses) - 1.0).clamp_min(0.0)


def _gradient_sensitivity(backbone: nn.Module, images: torch.Tensor,
                          lesion_masks: torch.Tensor) -> torch.Tensor:
    """G_l - ||dL/dW|| inside block l on a calibration batch.

    The proxy objective is "make the patch-token energy track lesion density",
    which is the only supervision available before any head exists.  Blocks the
    task wants to move produce a large gradient here and earn adapter capacity;
    blocks that already encode what is needed produce a small one.
    """
    was = {n: p.requires_grad for n, p in backbone.named_parameters()}
    for p in backbone.parameters():
        p.requires_grad_(True)
    backbone.zero_grad(set_to_none=True)

    feats: list[torch.Tensor] = []
    hooks = [blk.register_forward_hook(
        lambda _m, _i, o: feats.append(o if isinstance(o, torch.Tensor) else o[0]))
        for blk in _iter_blocks(backbone)]
    was_training = backbone.training
    backbone.eval()
    with torch.enable_grad():
        backbone.forward_features(images)
    for h in hooks:
        h.remove()

    npx = getattr(backbone, "num_prefix_tokens", 1)
    dens_full = lesion_masks.amax(1, keepdim=True).float()
    loss = images.new_zeros(())
    for f in feats:
        if f.dim() != 3:
            continue
        tok = f[:, npx:, :]
        g = _patch_grid(tok.shape[1])
        if g is None:
            continue
        energy = tok.pow(2).mean(-1)
        energy = (energy - energy.mean(1, keepdim=True)) / (energy.std(1, keepdim=True) + 1e-6)
        dens = F.adaptive_avg_pool2d(dens_full, (g, g)).flatten(1)
        dens = (dens - dens.mean(1, keepdim=True)) / (dens.std(1, keepdim=True) + 1e-6)
        loss = loss + F.mse_loss(energy, dens)
    if loss.requires_grad:
        loss.backward()

    scores = []
    for blk in _iter_blocks(backbone):
        tot = 0.0
        for p in blk.parameters():
            if p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
        scores.append(math.sqrt(tot))
    backbone.zero_grad(set_to_none=True)
    for n, p in backbone.named_parameters():
        p.requires_grad_(was[n])
    backbone.train(was_training)
    return torch.tensor(scores, device=images.device, dtype=torch.float32)


def _unit_norm(v: torch.Tensor) -> torch.Tensor:
    lo, hi = v.min(), v.max()
    if float(hi - lo) < 1e-9:
        return torch.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


def gla_importance(backbone: nn.Module, images: torch.Tensor,
                   lesion_masks: torch.Tensor, cfg: BackboneCfg) -> dict:
    """S_l = lam_g G_l + lam_l L_l + lam_a A_l, each term min-max normalised."""
    G = _unit_norm(_gradient_sensitivity(backbone, images, lesion_masks))
    L = _unit_norm(_lesion_relevance(backbone, images, lesion_masks))
    A = _unit_norm(_lesion_attention_mass(backbone, images, lesion_masks))
    wsum = cfg.lam_grad + cfg.lam_lesion + cfg.lam_attn
    S = (cfg.lam_grad * G + cfg.lam_lesion * L + cfg.lam_attn * A) / max(wsum, 1e-6)
    return {"S": S.clamp(0, 1), "G": G, "L": L, "A": A}


def ranks_from_importance(S: torch.Tensor, cfg: BackboneCfg) -> list[int]:
    """r_l = r_min + (r_max - r_min) * S_l, with the lesion-sensitive pin."""
    lo, hi = cfg.lora_rank_min, cfg.lora_rank_max
    out = []
    for v in S:
        v = float(v)
        r = hi if v >= cfg.lesion_layer_boost else lo + (hi - lo) * v
        out.append(int(max(lo, min(hi, round(r)))))
    return out


# --------------------------------------------------------------------------
# Real RETFound weight loading
# --------------------------------------------------------------------------
class RETFoundCheckpointError(RuntimeError):
    """Raised when genuine RETFound weights could not be loaded."""


def _resolve_checkpoint(ckpt: str, fallback: str | None = None) -> str:
    """Accepts a local path, a 'repo_id:filename', or a bare HF repo id.

    Tries `ckpt` first (a local path resolves with no network at all), then
    `fallback` if `ckpt` cannot be resolved. Previously `fallback`
    (BackboneCfg.retfound_fallback) was defined but never actually consulted
    here - a run with a missing/wrong local path failed outright instead of
    trying the configured HuggingFace source, silently defeating the
    "falls back to the HF mirror" comment on the config field.
    """
    from huggingface_hub import hf_hub_download
    errors = []
    for candidate in (ckpt, fallback):
        if not candidate:
            continue
        if os.path.exists(candidate):
            return candidate
        # a Windows absolute path (C:\Users\...) also "contains a colon", so
        # naively splitting on ":" here misparses it as repo_id="C",
        # filename="\Users\..." instead of correctly falling through (it
        # already failed the os.path.exists check above, e.g. because the
        # file just hasn't been downloaded yet - that's a real "not found",
        # not a malformed repo spec). A genuine "repo_id:filename" spec's
        # repo half always contains a "/" (owner/name) and is never a bare
        # single drive letter.
        looks_like_windows_path = len(candidate) > 1 and candidate[1] == ":"
        if (":" in candidate and not looks_like_windows_path
                and not candidate.startswith(("http://", "https://"))):
            repo, fname = candidate.rsplit(":", 1)
            try:
                return hf_hub_download(repo_id=repo, filename=fname)
            except Exception as e:                      # noqa: BLE001
                errors.append(f"{candidate}: {e}")
                continue
        for fname in ("RETFound_mae_natureCFP.pth", "RETFound_mae_meh.pth",
                      "RETFound_mae_shanghai.pth", "pytorch_model.bin"):
            try:
                return hf_hub_download(repo_id=candidate, filename=fname)
            except Exception:                       # noqa: BLE001 - try the next name
                continue
        errors.append(f"{candidate}: no matching file found under any known name")
    raise RETFoundCheckpointError(
        "could not resolve a RETFound checkpoint from "
        f"{[c for c in (ckpt, fallback) if c]}. If using the official "
        "YukunZhou/RETFound_* HuggingFace repos, they are GATED: register an "
        "account, request access on the repo page, then run "
        "`huggingface-cli login --token YOUR_TOKEN` before training. "
        f"Attempts: {errors}")


def load_retfound_weights(model: nn.Module, ckpt: str, require: bool = True,
                          fallback: str | None = None) -> dict:
    """Load genuine RETFound CFP weights into a timm ViT.

    Returns a report dict.  On `require=True` any failure - unresolvable file,
    zero matched tensors, a patch-embedding shape mismatch - raises rather than
    leaving the model on its random / ImageNet initialisation.
    """
    try:
        path = _resolve_checkpoint(ckpt, fallback)
    except Exception as e:                       # noqa: BLE001
        if require:
            raise RETFoundCheckpointError(
                f"RETFound checkpoint '{ckpt}' (fallback '{fallback}') "
                f"could not be fetched: {e}") from e
        return {"loaded": False, "reason": str(e), "path": None}

    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "module", "teacher"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    sd = {k.replace("module.", "").replace("backbone.", ""): v
          for k, v in sd.items() if isinstance(v, torch.Tensor)}
    # the MAE release keeps a decoder and a mask token that a ViT encoder has no
    # slot for; dropping them here keeps the "unexpected" count meaningful
    sd = {k: v for k, v in sd.items()
          if not k.startswith(("decoder_", "mask_token")) and k != "head.weight"
          and k != "head.bias"}

    # positional embedding is stored at the pretraining resolution
    tgt = model.state_dict()
    if "pos_embed" in sd and "pos_embed" in tgt and sd["pos_embed"].shape != tgt["pos_embed"].shape:
        from timm.layers import resample_abs_pos_embed
        npx = getattr(model, "num_prefix_tokens", 1)
        new_len = tgt["pos_embed"].shape[1] - npx
        g = int(math.isqrt(new_len))
        sd["pos_embed"] = resample_abs_pos_embed(
            sd["pos_embed"], new_size=[g, g], num_prefix_tokens=npx, verbose=False)

    matched = {k: v for k, v in sd.items()
               if k in tgt and tgt[k].shape == v.shape}
    shape_mismatch = sorted(k for k, v in sd.items()
                            if k in tgt and tgt[k].shape != v.shape)
    missing, unexpected = model.load_state_dict(matched, strict=False)
    n_backbone = sum(1 for k in tgt if k.startswith(("blocks.", "patch_embed.")))
    n_matched_backbone = sum(1 for k in matched if k.startswith(("blocks.", "patch_embed.")))
    coverage = n_matched_backbone / max(n_backbone, 1)

    report = {
        "loaded": True, "path": str(path), "source": ckpt,
        "matched_tensors": len(matched),
        "checkpoint_tensors": len(sd),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "shape_mismatched_keys": shape_mismatch,
        "backbone_coverage": round(coverage, 4),
        "missing_key_sample": list(missing)[:8],
        "unexpected_key_sample": list(unexpected)[:8],
    }
    if require and coverage < 0.95:
        raise RETFoundCheckpointError(
            f"RETFound load covered only {coverage:.1%} of the backbone tensors "
            f"({n_matched_backbone}/{n_backbone}). Refusing to train on a "
            f"partially-initialised backbone. Report: {report}")
    return report


# --------------------------------------------------------------------------
# Backbone wrapper
# --------------------------------------------------------------------------
@dataclass
class BackboneOutput:
    tokens: torch.Tensor      # (B, N, D) global patch tokens
    cls: torch.Tensor         # (B, D)
    grid: int
    local_tokens: torch.Tensor | None = None   # (B, C, n, D) per-crop tokens
    local_cls: torch.Tensor | None = None      # (B, C, D)
    local_grid: int = 0


class RETFoundPlusBackbone(nn.Module):
    """Real RETFound ViT, shared by the global image and the local crops.

    `dynamic_img_size` lets one weight set serve both resolutions: the position
    embedding is interpolated per call, so a 448 global view and 224 lesion
    crops go through the *same* parameters instead of needing two towers.
    """

    def __init__(self, cfg: BackboneCfg, img_size: int = 448):
        super().__init__()
        import timm
        self.cfg = cfg
        self._warned_resize_fallback = False
        self._resize_fallback_size = None
        self.vit = timm.create_model(cfg.name, pretrained=cfg.pretrained,
                                     num_classes=0, img_size=img_size,
                                     dynamic_img_size=True)
        if cfg.retfound_ckpt or cfg.retfound_fallback:
            self.load_report = load_retfound_weights(
                self.vit, cfg.retfound_ckpt, require=cfg.require_retfound,
                fallback=cfg.retfound_fallback)
            if self.load_report.get("loaded"):
                self.status = (f"RETFound weights loaded from {self.load_report['path']} "
                               f"(matched {self.load_report['matched_tensors']}, "
                               f"missing {self.load_report['missing_keys']}, "
                               f"unexpected {self.load_report['unexpected_keys']}, "
                               f"backbone coverage {self.load_report['backbone_coverage']:.1%})")
            else:
                # require_retfound=False and no checkpoint could be resolved at
                # all (as opposed to resolved-but-low-coverage, which still
                # raises inside load_retfound_weights) - load_report here is
                # just {"loaded": False, "reason": ..., "path": None}, so fall
                # back to ImageNet/random init exactly as the no-checkpoint
                # branch below does, instead of a KeyError on the report's
                # never-populated match/coverage fields.
                self.status = (f"RETFound weights NOT loaded "
                               f"({self.load_report.get('reason', 'unknown reason')}) - "
                               "falling back to " +
                               ("timm pretrained" if cfg.pretrained else "random init"))
        elif cfg.require_retfound:
            raise RETFoundCheckpointError(
                "backbone.require_retfound is set but both backbone.retfound_ckpt "
                "and backbone.retfound_fallback are empty. Point at least one at "
                "real RETFound weights, or explicitly clear require_retfound to "
                "run an ablation on a non-RETFound init.")
        else:
            self.load_report = {"loaded": False, "reason": "no checkpoint requested"}
            self.status = ("NO RETFound weights - "
                           + ("timm pretrained" if cfg.pretrained else "random init"))

        self.embed_dim = self.vit.num_features
        self.n_blocks = len(_iter_blocks(self.vit))
        if cfg.freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False
        self.lora_ranks: list[int] | None = None
        self.importance: dict | None = None
        self.n_unfrozen_blocks = 0

    # -- GLA-LoRA allocation ---------------------------------------------
    def fit_adapters(self, images: torch.Tensor, lesion_masks: torch.Tensor) -> dict:
        imp = gla_importance(self.vit, images, lesion_masks, self.cfg)
        ranks = ranks_from_importance(imp["S"], self.cfg)
        n_params = inject_lora(self.vit, ranks, self.cfg)
        self.lora_ranks = ranks
        self.importance = {k: [round(float(x), 4) for x in v] for k, v in imp.items()}
        return {"ranks": ranks, "lora_params": n_params,
                "importance_S": self.importance["S"],
                "grad_G": self.importance["G"],
                "lesion_L": self.importance["L"],
                "attn_A": self.importance["A"],
                "n_at_max_rank": sum(1 for r in ranks if r == self.cfg.lora_rank_max)}

    # -- staged fine-tuning ------------------------------------------------
    def set_stage(self, train_backbone: bool, train_lora: bool,
                  unfreeze_frac: float) -> dict:
        """Flip requires_grad to match one stage of the schedule."""
        blocks = list(_iter_blocks(self.vit))
        n_unfreeze = int(round(unfreeze_frac * len(blocks))) if train_backbone else 0
        first = len(blocks) - n_unfreeze

        for p in self.vit.parameters():
            p.requires_grad = False
        for i, blk in enumerate(blocks):
            if i >= first and n_unfreeze > 0:
                for n, p in blk.named_parameters():
                    # LoRA A/B are handled separately below
                    if ".A" not in n and ".B" not in n:
                        p.requires_grad = True
        if n_unfreeze >= len(blocks) and train_backbone:
            # near-full fine-tuning also releases the patch embedding and norm
            for mod in (getattr(self.vit, "patch_embed", None),
                        getattr(self.vit, "norm", None)):
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True
        for p in lora_parameters(self.vit):
            p.requires_grad = bool(train_lora)

        self.n_unfrozen_blocks = n_unfreeze
        return {"unfrozen_blocks": n_unfreeze, "total_blocks": len(blocks),
                "lora_trainable": bool(train_lora)}

    def param_groups(self) -> dict:
        """Split backbone params into the 'lora' and 'backbone' LR buckets."""
        lora_ids = {id(p) for p in lora_parameters(self.vit)}
        lora, base = [], []
        for p in self.vit.parameters():
            if not p.requires_grad:
                continue
            (lora if id(p) in lora_ids else base).append(p)
        return {"lora": lora, "backbone": base}

    # -- forward -----------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        try:
            f = self.vit.forward_features(x)
        except AssertionError as e:
            # DR-NOVA fix: guards a real reported crash -
            # "AssertionError: Input height (320) doesn't match model (224)".
            # dynamic_img_size=True is meant to make the patch embedding
            # accept the global view and the local crops at different
            # resolutions from one tower; if it doesn't take effect for some
            # reason (timm version behaviour, a reload path that lost the
            # flag, or a caller passing an unexpected native crop_size
            # instead of the resized crop_input), this used to crash the
            # whole run instead of degrading. This resizes ONLY on that
            # failure, to the size the patch embedding module was actually
            # built with, and warns once rather than silently masking a real
            # upstream sizing bug forever.
            if not self._warned_resize_fallback:
                expected = getattr(getattr(self.vit, "patch_embed", None),
                                   "img_size", (224, 224))
                print(f"[a4_backbone] WARNING: patch embedding rejected input "
                      f"size {tuple(x.shape[-2:])} ({e}). Falling back to a "
                      f"resize to {tuple(expected)} so the run continues, but "
                      f"this means dynamic_img_size did not take effect as "
                      f"intended for this call - the resolution benefit for "
                      f"this branch is lost until the real cause is fixed "
                      f"upstream (check crop_input vs. the size actually "
                      f"reaching the backbone).")
                self._warned_resize_fallback = True
                self._resize_fallback_size = tuple(expected)
            x = F.interpolate(x, size=self._resize_fallback_size,
                              mode="bilinear", align_corners=False)
            f = self.vit.forward_features(x)
        npx = getattr(self.vit, "num_prefix_tokens", 1)
        cls = f[:, 0] if npx > 0 else f.mean(1)
        tok = f[:, npx:] if npx > 0 else f
        return tok, cls, int(math.isqrt(tok.shape[1]))

    def forward(self, x: torch.Tensor, crops: torch.Tensor | None = None
                ) -> BackboneOutput:
        """x: (B,3,Hg,Wg) global view.  crops: (B,C,3,Hc,Wc) local lesion crops."""
        tok, cls, g = self._encode(x)
        if crops is None:
            return BackboneOutput(tokens=tok, cls=cls, grid=g)
        B, C = crops.shape[:2]
        ctok, ccls, cg = self._encode(crops.flatten(0, 1))
        return BackboneOutput(
            tokens=tok, cls=cls, grid=g,
            local_tokens=ctok.view(B, C, ctok.shape[1], ctok.shape[2]),
            local_cls=ccls.view(B, C, -1), local_grid=cg)
