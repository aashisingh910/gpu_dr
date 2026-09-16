"""Central configuration for the RETFound Plus-LAFT-XAI pipeline.

Every block (A1-A10) reads its hyper-parameters from here so that a single
`--config`-style override on the command line propagates everywhere.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
CKPT_DIR = DATA_DIR / "checkpoints"
OUT_DIR = PROJECT_ROOT / "outputs"


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- A1: Adaptive Quality & Domain Gate -------------------------------------
@dataclass
class QualityCfg:
    """Q* = w1*Q_quality + w2*Q_domain + w3*Q_lesion + w4*Q_blur + w5*Q_illum.

    The gate is no longer a single accept/reject on a blended quality number:
    each of the five factors is scored on its own axis, and the *routing* uses
    the pooled Q* to choose between retake, enhanced high-res, and the plain
    high-res pathway.
    """
    tau_r: float = 0.32          # Q* < tau_r  -> RETAKE
    tau_d: float = 0.60          # tau_r <= Q* < tau_d -> ENHANCE ; >= tau_d -> ACCEPT
    blur_ref: float = 220.0      # Laplacian variance that maps to a blur score of 1.0
    # weights of the five Q* terms (normalised at use time)
    weights: dict = field(default_factory=lambda: {
        "quality": 0.30, "domain": 0.10, "lesion": 0.25,
        "blur": 0.20, "illumination": 0.15,
    })
    # sub-weights inside the composite Q_quality term
    quality_sub: dict = field(default_factory=lambda: {
        "gradability": 0.50, "artifact": 0.28, "fov": 0.22,
    })
    # reference statistics of the training domain, used by Q_domain.  Filled in
    # by scripts/02_preprocess.py on a first pass and written back to the cache.
    domain_ref: dict = field(default_factory=lambda: {
        "rg_ratio": 1.95, "rg_sd": 0.45, "fill": 0.74, "fill_sd": 0.14,
        "sat": 0.42, "sat_sd": 0.16,
    })
    # Q_lesion maps the corpus's [p5, p85] lesion-CNR range onto [0, 1].  These
    # defaults suit the 224px gaussian-filtered EyePACS mirror; 02_preprocess
    # recalibrates them on whatever data it is pointed at and writes the result
    # to the cache, because the raw CNR scale is pipeline-dependent.
    lesion_floor_cnr: float = 6.0
    lesion_ref_cnr: float = 20.0


# --- A2: Adaptive Lesion-Preserving Preprocessing ---------------------------
@dataclass
class PreprocCfg:
    """Resolution policy.

    `cache_size` is the retinal-field crop written to disk - the "1024x1024
    global image" of the architecture.  The global ViT branch consumes a
    `global_size` resize of it; the local branch consumes `n_crops` windows of
    `crop_size` cut at *native* cache resolution and fed at `crop_input`.
    Local crops are what actually resolve microaneurysms; the global branch
    only has to supply context.
    """
    cache_size: int = 1024       # retinal field stored in the cache
    global_size: int = 448       # global branch input
    crop_size: int = 320         # window cut at cache resolution
    crop_input: int = 224        # window resized to this for the backbone
    n_crops: int = 6             # 4-8 local lesion crops
    crop_jitter: float = 0.12    # train-time positional jitter, fraction of crop
    clahe_clip: float = 2.5
    clahe_grid: int = 8
    illum_sigma: float = 25.0    # Gaussian sigma for background illumination estimate
    fusion_gamma: float = 0.6    # weight of enhanced image in the quality-aware fusion
    # --- learned ALPP fusion (torch side) ---
    alpp_enabled: bool = True
    alpp_gate_dim: int = 64
    alpp_branches: tuple = ("norm", "contrast", "vessel", "lesion")
    alpp_prior: tuple = (1.2, 0.6, 0.3, 0.3)   # gate bias init, favours I_norm
    # --- Objective 2 fix: soft exudates lose contrast under CLAHE -------
    # Evidence: paired CNR ratio for soft exudates is 0.945x (CI 0.897-1.039,
    # win rate 43%) - the only lesion class CLAHE makes WORSE.
    #
    # 'local_contrast' (a smooth division-normalisation, no tile boundaries)
    # was tried as a full CLAHE replacement and REJECTED after benchmarking:
    # at a sigma large enough to matter it pulls the local mean into a large
    # lesion's own interior exactly like a CLAHE tile does (soft-exudate CNR
    # measured 0.24-0.56 across sigma=8..60 on a synthetic blob, all WORSE
    # than doing nothing at 0.79), and it is 3-17x SLOWER than OpenCV's
    # tuned CLAHE implementation (60-310ms vs 17.6ms/image at 1024px,
    # measured). Kept only as an explicit, clearly-inferior ablation option
    # for anyone who wants to reproduce that comparison - do not default to
    # it. 'clahe' (default) + `clahe_protect_large_lesions` below is the
    # validated fix: detect large already-bright blobs and blend CLAHE's
    # output back toward native pixels there, which measurably restores the
    # blob's own contrast without touching CLAHE's (fast) global mechanism.
    enhancement_algorithm: str = "clahe"   # "clahe" (validated) | "local_contrast" (rejected, kept for comparison)
    local_contrast_sigma: float = 40.0
    local_contrast_strength: float = 0.8
    clahe_protect_large_lesions: bool = True
    large_lesion_kernel: int = 41        # structuring element scale for "large" blobs
    large_lesion_protect_strength: float = 0.6   # 0 = no protection, 1 = fully native
    # --- Objective 3 experiment: minimally-processed pixels -------------
    # When True, A2 caches field-extraction only (no illumination
    # normalisation, no CLAHE, no lesion-boost fusion). This is the "one
    # cheap experiment" from Objective 3's evidence pack: it tests whether
    # RETFound's frozen-feature disadvantage vs generic ImageNet backbones
    # is really about retinal domain knowledge, or about A1/A2 preprocessing
    # not matching RETFound's own pretraining statistics.
    raw_pixels: bool = False


# --- A3: Lesion Specialist Mixture-of-Experts -------------------------------
@dataclass
class MoECfg:
    experts: tuple = ("MA", "HE", "EX_H", "EX_S", "NV", "ME")
    expert_dim: int = 128
    lesion_dim: int = 256
    mask_size: int = 112         # resolution of the lesion evidence maps
    gate_hidden: int = 128
    gate_temperature: float = 1.0
    gate_context: tuple = ("global", "lesion", "quality", "hardness")
    # --- novelty fix: route the local crops through the lesion MoE too ---
    # Previously the MoE only ever saw the 448px GLOBAL view (moe_in in
    # model.py); the local crops were used by the backbone but never by the
    # lesion evidence pathway. Objective 2's own evidence pack shows
    # resolution - not enhancement - is what makes microaneurysms visible at
    # all (a whole-image 224px view puts 100% of annotated MAs below the
    # noise floor; a 320px native-resolution crop recovers 61% of them).
    # Running the SAME MoE weights on each crop and pooling presence with the
    # global pass by elementwise max ("a lesion the global view missed but a
    # crop caught should not be washed out") uses the local branch for
    # exactly what it exists to provide, at no new parameter cost.
    use_crop_evidence: bool = True


# --- A4: RETFound Plus + Gradient-Lesion Adaptive LoRA (GLA-LoRA) -----------
@dataclass
class BackboneCfg:
    name: str = "vit_large_patch16_224"
    # Real RETFound CFP weights. Either a local .pth/.bin or "repo_id:filename".
    # Local path is tried first (fast, no network); falls back to
    # `retfound_fallback` below if the local copy is absent.
    retfound_ckpt: str | None = str(CKPT_DIR / "RETFound_MAE" / "pytorch_model.bin")
    # Official source (Zhou et al. 2023, Nature): rmaphoh/RETFound /
    # YukunZhou's HuggingFace repos. This is GATED - register a HuggingFace
    # account, request access on the repo page, then
    # `huggingface-cli login --token YOUR_TOKEN` before the first run. The
    # CFP (colour fundus photo) checkpoint is the correct variant for
    # EyePACS/DR - NOT the OCT one, which is trained on a different modality.
    retfound_fallback: str = "YukunZhou/RETFound_mae_natureCFP:RETFound_mae_natureCFP.pth"
    require_retfound: bool = True   # hard-fail instead of silently using ImageNet
    pretrained: bool = False        # never pull timm/ImageNet weights by default
    freeze_backbone: bool = True    # stage 1; staged schedule unfreezes later
    share_local_backbone: bool = True   # local crops reuse the global weights
    lora_rank_min: int = 2
    lora_rank_max: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_targets: tuple = ("qkv", "proj")
    # GLA-LoRA importance:  S_l = lam_g*G_l + lam_l*L_l + lam_a*A_l
    lam_grad: float = 0.35       # gradient sensitivity of the block
    lam_lesion: float = 0.45     # token-energy / lesion-density correlation
    lam_attn: float = 0.20       # attention mass landing on lesion patches
    lesion_layer_boost: float = 0.85  # S_l above this -> forced to r_max


# --- A5: Bidirectional Pathology-Anatomy Fusion -----------------------------
@dataclass
class FusionCfg:
    n_heads: int = 8
    anatomy_regions: tuple = ("optic_disc", "macula", "sup_arcade",
                              "inf_arcade", "peripheral")
    gnn_layers: int = 2
    knn: int = 6
    fused_dim: int = 384
    bidirectional: bool = True
    probe_dim: int = 128         # dimension of the shared P(.) consistency probe


# --- A6/A7/A8 heads ---------------------------------------------------------
@dataclass
class HeadCfg:
    n_grades: int = 5
    horizons: tuple = (6, 12, 24, 60)      # months
    clinical_dim: int = 8
    dropout: float = 0.1
    # hierarchical binary tasks, each a threshold on the ordinal scale
    hierarchy: tuple = ("any_dr", "referable", "stdr", "pdr")
    hierarchy_thresholds: tuple = (1, 2, 3, 4)
    proj_dim: int = 128          # supervised-contrastive projection head


# --- A9/A10 -----------------------------------------------------------------
@dataclass
class XaiCfg:
    counterfactual_blur: int = 25
    # lambda_XAI is 0 while the high-resolution / lesion-pretrained encoder
    # settles, then ramps.  Index by stage (1-based); stage 1 gets schedule[0].
    weight_schedule: tuple = (0.0, 0.0, 0.02, 0.05)
    ramp_within_stage: bool = True     # linearly ramp to the stage's value
    grounding_topk: int = 3            # lesion classes cited per explanation


@dataclass
class ScreeningCfg:
    """Referral-triage objective.

    The grader is scored as a screening service, not as a grade-namer: pick the
    threshold that clears the most patients while keeping the cleared bucket at
    least `npv_target` clean, and use the cleared fraction as the number that
    decides which checkpoint wins.  `auc_tiebreak` folds a little of the ranking
    quality back in so that two checkpoints which clear the same fraction are
    separated by the one with the better-ordered scores, and so that a model
    which cannot reach the constraint at all is still ranked sensibly.
    """
    task_grade: int = 3              # sight-threatening DR, grade >= 3
    npv_target: float = 0.985
    auc_tiebreak: float = 0.25


@dataclass
class UncertaintyCfg:
    mc_samples: int = 10
    entropy_hi: float = 0.85               # normalised entropy above this = "high"
    calibrator: str = "temperature"        # temperature | isotonic | vector


# --- losses -----------------------------------------------------------------
@dataclass
class LossCfg:
    """L = L_ord + l1*L_hier + l2*L_bound + l3*L_contr + l4*L_les + l5*L_PA + l6*L_XAI

    When `learned_weights` is True, w_ordinal..w_cbf are IGNORED and replaced
    by a HomoscedasticLossWeighting module that learns one log-variance per
    term instead (Kendall et al. 2018) - see src/dr/modules/loss_weighting.py.
    The fixed values below remain as the reproducible/legacy fallback and as
    the sensible init point the learned weights start from (log_var=0 ==
    weight=1, not these specific numbers - they matter only when
    learned_weights=False).
    """
    learned_weights: bool = True
    # --- DR-NOVA addition: attack QWK directly, not just via proxies -------
    # Every other term here is a proxy for the quadratic-weighted kappa the
    # whole system is judged on; none of them shares its exact reward shape.
    # See dqk_loss (src/dr/modules/a6_ordinal.py) for the formula and its
    # relationship to de la Torre et al. 2018's weighted-kappa loss family.
    use_dqk_loss: bool = True
    w_dqk: float = 0.5     # used only when learned_weights=False (fixed fallback)
    w_ordinal: float = 1.0
    w_hierarchical: float = 0.5
    w_boundary: float = 0.2
    w_contrastive: float = 0.2
    w_lesion: float = 0.3
    w_pa: float = 0.1
    # XAI weight comes from XaiCfg.weight_schedule (0 -> 0.02 -> 0.05)
    w_cbf: float = 0.3        # class-balanced focal, kept as a moderate auxiliary
    focal_gamma: float = 2.0
    cb_beta: float = 0.999
    boundary_gamma: float = 1.5   # L_boundary = |y - yhat|^gamma
    contrastive_temp: float = 0.1


# --- sampling ---------------------------------------------------------------
@dataclass
class SamplingCfg:
    """Moderate re-balancing + hard-example mining.

    P_i  proportional to  n_class^power  *  (1 + H_i)^eta
    with H_i = L_i / (mean L + eps) refreshed after every epoch.
    """
    power: float = -0.5          # n^-0.5, gentler than full inverse-frequency
    hard_eta: float = 0.5
    hard_eps: float = 1e-3
    hard_ema: float = 0.7        # EMA over epochs, so one noisy loss cannot spike
    hard_clip: float = 4.0       # cap on (1+H)^eta
    boundary_bonus: float = 0.6  # extra weight for adjacent-class confusions
    lowconf_bonus: float = 0.4   # extra weight for low-margin predictions
    class_weight_power: float = -0.25   # moderate loss weighting on top


# --- staged fine-tuning -----------------------------------------------------
@dataclass
class StageCfg:
    name: str
    epochs: int
    train_backbone: bool         # any backbone grad at all
    train_lora: bool
    unfreeze_frac: float         # fraction of the *last* transformer blocks unfrozen
    head_lr: float
    lora_lr: float
    backbone_lr: float
    # A2 ALPP sits *in front of* the backbone, so any gradient it needs must
    # traverse every transformer block.  While the backbone is fully frozen and
    # no LoRA is active, that is a full-depth backward pass bought purely to
    # update ~57k preprocessing weights: measured on a ViT-Large it turns a
    # 362 ms step into 3322 ms, a 9.2x penalty.  From stage 2 the LoRA adapters
    # already force the same full-depth backward, so training ALPP there is
    # free.  Hence: off in stage 1, on afterwards.
    train_alpp: bool = True


def default_stages() -> tuple[StageCfg, ...]:
    return (
        StageCfg("s1_frozen",    5,  False, False, 0.00, 1e-4, 0.0,  0.0, False),
        StageCfg("s2_lora",     12,  False, True,  0.00, 1e-4, 5e-5, 0.0, True),
        StageCfg("s3_unfreeze", 14,  True,  True,  0.33, 5e-5, 5e-5, 5e-6, True),
        StageCfg("s4_full",     14,  True,  True,  1.00, 2e-5, 2e-5, 5e-6, True),
    )


# --- training ---------------------------------------------------------------
@dataclass
class TrainCfg:
    stages: tuple = field(default_factory=default_stages)
    batch_size: int = 8
    weight_decay: float = 0.02
    warmup_frac: float = 0.1
    # 0 by default: the cache is a uint8 memmap that costs ~2 ms/image to read
    # against ~80 ms/image of compute, so worker processes buy nothing - and on
    # macOS forked workers deadlock against the open memmap + MPS context.
    # Override with --workers on Linux/CUDA if you want prefetch overlap.
    num_workers: int = 0
    seed: int = 1337
    max_train: int | None = None   # subsample the training split (None = all)
    max_val: int | None = None
    samples_per_epoch: int | None = None
    grad_clip: float = 1.0
    grad_accum: int = 1
    # "ordinal"   -> 0.5*QWK + 0.3*MacroF1 + 0.2*minority recall
    # "screening" -> fraction cleared at the NPV constraint (see ScreeningCfg)
    select_objective: str = "ordinal"
    # early stopping on 0.5*QWK + 0.3*MacroF1 + 0.2*minority recall
    select_weights: dict = field(default_factory=lambda: {
        "qwk": 0.5, "macro_f1": 0.3, "minority_recall": 0.2})
    minority_grades: tuple = (1, 2, 3)     # Mild / Moderate / Severe
    patience: int = 8                      # epochs without selection-score gain
    # lesion pretraining (IDRiD / DDR / FGADR) before EyePACS grading
    lesion_pretrain_epochs: int = 8
    lesion_pretrain_lr: float = 3e-4
    # --- Objective 1 fix: the CORAL decision ladder ---------------------
    # Evidence: the learned thresholds match their random initialisation to
    # 4 decimal places after a full run (drift ~1e-4), which is why Mild and
    # Severe recall is pinned at exactly zero regardless of sampling/loss
    # weighting - reachability is a property of these frozen parameters, not
    # of the data. Two independent fixes, on by default:
    init_ladder_from_prior: bool = True   # seed b_k at logit(P(Y>k)) on train counts
    ladder_lr_mult: float = 10.0          # ordinal.b0/deltas train at head_lr * this
    # --- efficiency/stability additions -----------------------------------
    amp: bool = False          # MPS autocast is flaky on some ops; off by default there.
                                # NOW ACTUALLY WIRED UP in 03_train.py (previously dead
                                # config - defined here but never read anywhere). Safe to
                                # turn on for cuda/cpu; stays off on mps regardless.
    ema_decay: float = 0.995   # 0 disables. Tuned for FEW total optimizer steps (a
                                # small dataset x grad_accum means only ~1-3k steps per
                                # run) - the usual 0.999/0.9999 seen in large-batch,
                                # many-step training would barely move from init here.


@dataclass
class DomainAdaptCfg:
    """A11 - DAFA: Domain-Adversarial Feature Alignment.

    DR-NOVA addition, attacking the domain-generalisation weakness measured
    directly (EyePACS QWK 0.5944 -> APTOS QWK 0.3088). See
    src/dr/modules/a11_domain_adapt.py for the full mechanism. A complete
    no-op unless the training loop supplies `batch["domain_id"]` - present
    but idle when training on a single labelled corpus.
    """
    enabled: bool = True          # module exists and computes domain_logits;
                                   # only affects gradients when domain_id is supplied
    n_domains: int = 2            # 0 = in-domain (EyePACS), 1 = external (e.g. APTOS), ...
    hidden: int = 128
    dropout: float = 0.1
    grl_gamma: float = 10.0       # DANN's standard lambda-ramp steepness
    loss_weight: float = 0.1      # kept fixed (not folded into learned_weights) -
                                   # this term should not compete for scale with the
                                   # single-domain grading losses when domain_id is absent


@dataclass
class Config:
    quality: QualityCfg = field(default_factory=QualityCfg)
    preproc: PreprocCfg = field(default_factory=PreprocCfg)
    moe: MoECfg = field(default_factory=MoECfg)
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    head: HeadCfg = field(default_factory=HeadCfg)
    xai: XaiCfg = field(default_factory=XaiCfg)
    uncertainty: UncertaintyCfg = field(default_factory=UncertaintyCfg)
    screening: ScreeningCfg = field(default_factory=ScreeningCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    domain_adapt: DomainAdaptCfg = field(default_factory=DomainAdaptCfg)
    device: str = "auto"
    # component ablations for the model-comparison study; any of
    # {"moe", "fusion", "ordinal", "clinical", "xai", "local", "alpp",
    #  "bidirectional", "hierarchical", "crop_moe", "lesion_gate", "domain_adapt"}
    ablate: tuple = ()

    def torch_device(self) -> torch.device:
        return pick_device(self.device)

    def total_epochs(self) -> int:
        return sum(s.epochs for s in self.train.stages)

    def save(self, path: os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=list))


def default_config() -> Config:
    return Config()
