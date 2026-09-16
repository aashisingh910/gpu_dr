# RETFound Plus–LAFT-XAI

**Lesion-Aware Adaptive Foundation Transfer with Temporal Prognosis and Explainable AI
for Diabetic Retinopathy**

A complete, runnable implementation of the ten-block architecture (A1–A10, see
[The model in one page](#the-model-in-one-page)), trained and evaluated on the
**real EyePACS 2015** dataset (35,126 clinician-graded fundus photographs,
Kaggle Diabetic Retinopathy Detection).

> **What is not in this repo:** the ~10 GB dataset and cache, and the trained
> weights (`best.pt` is 124 MB, above GitHub's 100 MB file limit). Both are
> reproducible from the scripts — see [Steps to run](#steps-to-run). The result
> JSONs in `outputs/` *are* tracked, so every number quoted below can be checked
> without retraining.

> Research prototype. **Not a medical device. Not for clinical use.**

---

## Headline result — referral-safety triage

> ⚠️ **The numbers in this section are the PRE-CHANGE baseline**: 224 px input,
> ImageNet-initialised ViT-Small, frozen backbone + fixed-rank LoRA, 4 epochs,
> morphological pseudo-lesions. They are kept because they are the thing the
> current architecture is being measured against. See
> [Results after the resolution / RETFound / staged-training changes](#results-after-the-changes)
> for the current run.

The system is best used as a **screening triage** that decides who can safely be
cleared, not as a grade-namer. On the held-out EyePACS test split (5,268 images,
2,634 patients, patient-disjoint), for **sight-threatening DR (grade ≥ 3)**:

| safety target | thr | sens | spec | **NPV** | refer % | **cleared %** | missed |
|---|---|---|---|---|---|---|---|
| sensitivity ≥ 0.90 | 0.104 | 0.902 | 0.801 | **0.9943** | 23.0% | **77.0%** | 23 |
| sensitivity ≥ 0.95 | 0.074 | 0.919 | 0.717 | **0.9948** | 31.1% | 68.9% | 19 |
| NPV ≥ 0.99 | 0.185 | 0.838 | 0.902 | **0.9917** | 13.1% | **86.9%** | 38 |

**99.4% negative predictive value while removing 77% of clinician workload.**

Thresholds are selected on the **validation** split and applied unchanged to
test (`scripts/09_screening.py`). This is why the `NPV ≥ 0.98` target lands at
0.9799 on test rather than exactly 0.98 — tuning the threshold on the data you
report it on inflates the number and is the most common way a screening result
becomes meaningless.

**Limits you must respect:**
- **Referable DR (≥ 2) is not viable as triage here** — reaching NPV 0.98
  requires referring 95.1% of patients, clearing only 4.9%.
- **Thresholds do not transfer.** On the external APTOS cohort the same cut
  gives NPV 0.9155 and misses 253 cases. Every new site needs its own
  threshold recalibration on local validation data.

### 5-class grading (the underlying task)

| metric | value |
|---|---|
| Accuracy | 0.7646 (majority baseline 0.7380) |
| Quadratic weighted kappa | 0.5944 |
| Adjacent accuracy (±1) | 0.8517 |
| Sight-threatening AUC | 0.9373 |
| Referable-DR AUC | 0.8518 |

Per-grade recall is **No DR 0.977, Mild 0.000, Moderate 0.187, Severe 0.055,
PDR 0.694**. Mild DR is never detected: grade 1 is defined by microaneurysms,
which are ~1 pixel at 224 px in a mirror that has additionally been
gaussian-filtered. That diagnosis — a resolution limit rather than an
architecture limit — is what motivated the 1024 px cache and the local lesion
crops described below.

Note that ~98% *5-class* accuracy is not achievable on EyePACS by any model:
it would require predicting the labels more consistently than the clinicians
who wrote them. Published state of the art is ≈0.84–0.85 QWK.

---

## The model in one page

A single fundus photograph enters, a referral decision leaves. The chain is:

```
raw fundus (any size)
  │
  ├─ A1  Q* gate ................. 5 independent axes → Q* → retake / enhance / accept
  │        Q* = w1·Q_quality + w2·Q_domain + w3·Q_lesion + w4·Q_blur + w5·Q_illum
  │        Q_lesion is the one that matters: lesion-scale contrast-to-noise, i.e.
  │        "could a microaneurysm survive this image at all"
  ├─ A2  field extraction ........ FOV crop → 1024×1024 retinal field
  │        Q* medium → high-resolution enhancement pathway; Q* high → native pixels
  │      (A1/A2 are OpenCV on CPU, cached once by step 02, together with the
  │       centres of the local lesion windows)
  ▼
  1024×1024 cached field
  ├──────────────────────────────┬─────────────────────────────┐
  ▼                              ▼                             │
global resize 448×448      6 local crops, 320 px cut           │
                           at native resolution, fed at 224    │
  └──────────────┬───────────────┘                             │
                 ▼                                             │
    A2 ALPP (learned, differentiable)                          │
    I* = w1·I_norm + w2·I_contrast + w3·I_vessel + w4·I_lesion  │
    [w1..w4] = Softmax(G(I))  — per image, not per dataset      │
                 │                                             │
  ┌──────────────┴──────────────┐                              │
  ▼                             ▼                              ▼
A3 Lesion MoE              A4 real RETFound ViT-L/16      (crop centres from
6 experts, adaptive gate   + GLA-LoRA adapters             the lesion priors)
α = Softmax(G(Z_G,Z_L,Q,H))  shared by global + local
  │  Z_L, evidence            │  Z_G tokens, per-crop CLS
  └───────────┬───────────────┘
              ▼
        A5 fusion:  X-Attn(global↔lesion) → BIDIRECTIONAL pathology↔anatomy
                    Z_P→A = CA(Z_P, Z_A) ;  Z_A→P = CA(Z_A, Z_P)
                    Z_F = Fusion(Z_P, Z_A, Z_P→A, Z_A→P, G)
                    → graph (kNN + spatial + lesion-in-region edges) → GNN
              ▼
        A8 clinical fusion (gated; inert without metadata) → Z_MC
              ▼
   ┌──────────┼────────────┬──────────────┐
   ▼          ▼            ▼              ▼
A6 hierarchical  A7 temporal  A9 lesion-grounded  A10 uncertainty
ordinal CORAL    (needs        XAI chain           MC-dropout, temperature
+ any-DR/refer/  longitudinal) prediction→attention scaling, decision gate
  STDR/PDR heads               →lesion→anatomy
+ boundary                     →severity→confidence
+ contrastive
   │
   ▼
Screening triage → REFER / REVIEW / CLEAR   (thresholds from step 09)
```

**Resolution.** The cache holds a 1024×1024 retinal field. The global branch
sees a 448 px resize of it for context; the local branch cuts six 320 px windows
at *native* cache resolution and feeds them at 224 px. A window taken this way
shows a microaneurysm at roughly 4.5× the sampling density of the same lesion
inside a 224 px whole-image view, which is the entire reason the change exists —
Mild and Moderate are defined by lesions that a 224 px downsample deletes.
Crop placement is not a grid: the macula and optic disc are always included, the
rest go to non-max-suppressed peaks of the lesion prior.

**Backbone.** Real RETFound MAE ViT-Large/16 (`vit_large_patch16_224`, 1024-dim,
24 blocks), loaded from the published CFP checkpoint. The loader reports matched
/ missing / unexpected tensors and **refuses to train** if backbone coverage
falls below 95% — there is no silent ImageNet fallback, because a run that
quietly trains an `augreg_in21k` ViT and reports it as RETFound is worse than a
run that fails. `--allow-no-retfound` exists only for the ablation arm.
`dynamic_img_size` lets the same weights serve the 448 px global view and the
224 px crops, so there is one tower, not two.

**Staged fine-tuning.** Frozen → adapters → partial unfreeze → near-full, with
three learning rates live at once:

| stage | epochs | backbone | LoRA | head LR | LoRA LR | backbone LR |
|-------|--------|----------|------|---------|---------|-------------|
| 1 `s1_frozen`    | 5  | frozen            | off | 1e-4 | –    | –    |
| 2 `s2_lora`      | 12 | frozen            | on  | 1e-4 | 5e-5 | –    |
| 3 `s3_unfreeze`  | 14 | last 33% of blocks| on  | 5e-5 | 5e-5 | 5e-6 |
| 4 `s4_full`      | 14 | all blocks + patch embed | on | 2e-5 | 2e-5 | 5e-6 |

**GLA-LoRA — Gradient–Lesion Adaptive LoRA (A4).** Rank allocation now pools
three signals instead of one correlation:

```
S_l = 0.35·G_l + 0.45·L_l + 0.20·A_l
      G_l  ‖dL/dW‖ inside block l on a calibration batch  (what the task wants to move)
      L_l  |corr(patch-token energy, lesion density)|     (what tracks lesions)
      A_l  share of block l's attention landing on lesion patches
r_l = r_min + (r_max − r_min)·S_l ,  with S_l ≥ 0.85 pinned to r_max
```

**Lesion supervision is real, not morphological.** The A3 experts are pretrained
on 838 images with ophthalmologist pixel masks — IDRiD (81) and DDR (757),
covering MA / HE / hard exudates / soft exudates — then the stem and experts
transfer into EyePACS grading. Channels a dataset does not annotate (NV, ME) are
*masked out* of the loss rather than treated as negative; training an NV expert
against an all-zero target would teach it that neovascularisation never exists.

**Training objective.**
```
L = 1.0 ·L_ordinal       CORAL rank loss on the monotone cumulative head
  + 0.5 ·L_hierarchical  any-DR / referable / STDR / PDR, each its own head
  + 0.2 ·L_boundary      |y − ŷ|^1.5 on the continuous expected grade
  + 0.2 ·L_contrastive   supervised contrastive, weighted by ordinal distance
  + 0.3 ·L_lesion        A3 expert supervision
  + 0.1 ·L_PA            ‖P(Z_P) − P(Z_A)‖²  pathology/anatomy agreement
  + λ(t)·L_XAI           1 − Dice(attribution, lesion mask), 0 → 0.02 → 0.05
  + 0.3 ·L_CBF           class-balanced focal, a moderate auxiliary
```
λ_XAI is held at **zero** through stages 1–2. Tying attribution to lesion masks
while the attention map is still noise only teaches the model to match noise.

**Class imbalance.** Sampling is `P_i ∝ n_class^-0.5 · min((1+H_i)^0.5, 4)`, with
moderate `n^-0.25` loss weighting — not full inverse-frequency sampling stacked
on full class-balanced weighting, which over-corrects until grade 4 dominates
the gradient and the middle grades collapse. `H_i = L_i / mean(L)` is refreshed
after every epoch as an EMA, with extra weight for adjacent-class errors and
low-margin predictions, so the sampler pulls hardest exactly on the
No-DR↔Mild↔Moderate↔Severe↔PDR boundaries.

**Model selection** is `0.5·QWK + 0.3·MacroF1 + 0.2·minority recall`, never
accuracy: predicting "No DR" for everything scores 0.73 accuracy on EyePACS.

**Outputs per image:** DR grade 0–4 with a monotone ordinal distribution, four
hierarchical clinical probabilities, six lesion presences with gate weights,
predictive entropy split into aleatoric and epistemic, a grounded explanation
citing lesion→region→severity contributions with a chain-confidence score, a
counterfactual ΔP, and a REFER/REVIEW/CLEAR triage decision.

---

<a name="results-after-the-changes"></a>
## Results after the changes

Run configuration actually executed on this machine (8 GB Apple M3, MPS):

| | before | after |
|---|---|---|
| source data | `eyepacs-224`, gaussian-filtered 224 px | `eyepacs-hi`, native ~1024 px retinal field |
| input | one 224 px view | 224 px global view + 6 local crops cut at 1024 px |
| sampling density per lesion | 0.22 output px per field px | 0.70 (3.2x linear, 10x areal) |
| backbone | ViT-Small, ImageNet init | **real RETFound MAE ViT-Large/16**, 100% tensor coverage |
| adaptation | frozen + fixed-rank LoRA | GLA-LoRA + 4-stage frozen→LoRA→unfreeze→near-full |
| lesion supervision | morphological pseudo-labels | **838 real IDRiD + DDR masks**, then transfer |
| preprocessing | one fixed recipe | learned per-image ALPP fusion weights |
| fusion | pathology → anatomy | bidirectional + L_PA consistency |
| objective | CBF + ordinal + lesion + XAI | ordinal + hierarchical + boundary + contrastive + lesion + PA + scheduled XAI |
| imbalance | n^-1 sampling **and** full CB weighting | n^-0.5 + hard-example mining, moderate weighting |
| selection | best val QWK | 0.5·QWK + 0.3·MacroF1 + 0.2·minority recall |

The numbers this run produced are written to `outputs/retfound_plus_laft_xai/`
(`history.json`, `evaluation.json`, `gla_lora.json`) and
`outputs/lesion_pretrain/history.json`. See
[What is real, and what is not](#what-is-real-and-what-is-not) before quoting
any of them.

### Scale honesty

The published schedule in `config.py` is 5/12/14/14 epochs at 6,000 samples per
epoch over the full 35 k images. Driving a ViT-Large over one global plus six
local views costs ~14 backbone passes per sample; on this laptop that schedule
is roughly 40 GPU-hours, and a 1024 px cache of all 35 k images is 110 GB
against 8 GB of RAM. The run here therefore uses a stratified, patient-grouped
6,000-image subset and a shortened schedule. Every architectural change is
exercised in full — nothing is stubbed — but the epoch and data budget is
smaller than the configuration the code defaults to. On a CUDA box, raise
`--limit`, `--stage-epochs` and `--samples-per-epoch`; nothing else changes.

---

## Repository layout

### Core library — `src/dr/`

| file | lines | what it does |
|---|---|---|
| [`config.py`](src/dr/config.py) | 150 | Every hyper-parameter as dataclasses (`QualityCfg`, `BackboneCfg`, …), device selection (`cuda`/`mps`/`cpu`), and the `ablate` switch used by the comparison study. |
| [`model.py`](src/dr/model.py) | 120 | Assembles A3→A10 into `RETFoundPlusLAFTXAI`. `fit_adapters()` runs the A4 rank calibration; `forward()` returns every block's output in one dict. |
| [`metrics.py`](src/dr/metrics.py) | 250 | All 38 evaluation parameters in 7 groups, as pure-numpy functions so any model's outputs can be scored. |
| [`screening.py`](src/dr/screening.py) | 117 | Referral operating points: threshold sweep, constraint-based selection (sensitivity or NPV floor), workload-reduction accounting, `triage()` three-way decision. |
| [`report.py`](src/dr/report.py) | 150 | The Final Clinical AI Report — text rendering plus the 4-panel XAI figure (original / attribution / lesion evidence / counterfactual). |

### Architecture blocks — `src/dr/modules/`

| file | lines | what it does |
|---|---|---|
| [`a1_quality.py`](src/dr/modules/a1_quality.py) | 198 | Gradability, blur (Laplacian variance), illumination (exposure + quadrant drift), artefacts (blowout/dust), FOV completeness → weighted score Q and the retake/enhance/accept decision. Also the camera-domain fingerprint. |
| [`a2_preprocess.py`](src/dr/modules/a2_preprocess.py) | 159 | Retinal field extraction, Graham illumination normalisation, CLAHE whose clip limit *rises as quality falls*, multi-scale vessel and lesion enhancement, and quality-aware fusion `α = γ(1−Q)`. |
| [`a3_lesion_moe.py`](src/dr/modules/a3_lesion_moe.py) | 139 | Six dilated-conv experts (MA/HE/EX-H/EX-S/NV/ME) with per-lesion receptive fields, a softmax gate, and a Dice+weighted-BCE supervision loss. |
| [`a4_backbone_lora.py`](src/dr/modules/a4_backbone_lora.py) | 430 | `LoRALinear`, `gla_importance()` (gradient + lesion + attention signals), `ranks_from_importance()`, `inject_lora()`, `set_stage()` for the four-stage schedule, and the hard-failing RETFound loader. |
| [`a5_fusion_gnn.py`](src/dr/modules/a5_fusion_gnn.py) | 214 | Two cross-attention stages, anatomy tokens pooled from real region maps, three-edge-type graph construction, and a dense-adjacency GAT (no torch-geometric dependency). |
| [`a6_ordinal.py`](src/dr/modules/a6_ordinal.py) | 105 | CORAL head with structurally monotone thresholds, rank-target encoding, and class-balanced focal loss. |
| [`a7_temporal.py`](src/dr/modules/a7_temporal.py) | 117 | Time-embedded causal transformer over visits, per-horizon risk + discrete-time survival head. **Inert on EyePACS** (no follow-up visits). |
| [`a8_multimodal.py`](src/dr/modules/a8_multimodal.py) | 69 | Clinical MLP encoder + gated cross-attention. The gate is hard-zeroed when metadata is absent, so the model degrades to image-only rather than consuming imputed zeros. |
| [`a9_xai.py`](src/dr/modules/a9_xai.py) | 203 | Differentiable attribution for the training loss, Grad-CAM and attention rollout for reporting, counterfactual lesion inpainting, Dice and pointing-game metrics. |
| [`a10_uncertainty.py`](src/dr/modules/a10_uncertainty.py) | 169 | MC-dropout ensembling with aleatoric/epistemic decomposition, temperature/vector/isotonic calibrators, and the clinical decision gate. |

### Data — `src/dr/data/`

| file | lines | what it does |
|---|---|---|
| [`eyepacs.py`](src/dr/data/eyepacs.py) | 176 | Manifest building for both EyePACS layouts, **patient-level** stratified splitting, and the memmap-backed `CachedEyePACS` dataset. |
| [`lesion_priors.py`](src/dr/data/lesion_priors.py) | 252 | The six morphological lesion detectors and the five anatomical region maps (optic disc localisation, macula from disc geometry, arcades, periphery). **Weak labels — read the honesty section.** |

### Scripts — `scripts/`

Numbered in execution order; each is standalone and re-runnable.

---

## Steps to run

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Kaggle credentials are needed for step 01 — `~/.kaggle/kaggle.json`, or
`KAGGLE_USERNAME` / `KAGGLE_KEY` in the environment.

### Step 0 — self-test (seconds)
```bash
.venv/bin/python scripts/08_selftest.py
```
Exercises all ten blocks on synthetic tensors plus a generated fundus, including
the A7 temporal path that EyePACS cannot train. Verifies ordinal monotonicity,
survival monotonicity, rank allocation bounds, and that gradient reaches every
trainable tensor. **Run this first** — it catches architecture bugs in seconds
instead of an hour into training. Expect `13/13 blocks passed`.

### Step 1a — download real EyePACS at 1024 px (~1 h, 7.8 GB)
```bash
.venv/bin/python scripts/01_download_data.py --dataset eyepacs-hi
```
Streams the Kaggle archive with resume support (the official CLI stalls behind
some proxies) and extracts with path-traversal protection.

**Use `eyepacs-hi`, not `eyepacs-224`.** The 224 px mirror is gaussian-filtered
and pre-downsampled: a microaneurysm is a handful of pixels at acquisition
resolution and simply is not present in those files. Upsampling them to 1024
recovers nothing — the information is gone. Every resolution claim in this
README assumes the 1024 px mirror; running the pipeline on `eyepacs-224` will
execute end to end and produce a global/local architecture reading interpolated
pixels, which is not the same thing.

### Step 1b — download real lesion annotations (~10 min, 1.1 GB)
```bash
.venv/bin/python scripts/11_download_lesions.py
```
Fetches IDRiD (81 images) and DDR (757) with pixel-level MA / HE / hard-exudate /
soft-exudate masks into `data/raw/lesion/`. FGADR adds neovascularisation and
IRMA but requires a signed request form from its authors; drop it at
`data/raw/lesion/fgadr/` and it is picked up automatically.

### Step 2 — Q\* gate + field extraction + lesion crops (~45 min, 7 workers)
```bash
.venv/bin/python scripts/02_preprocess.py --root data/raw/eyepacs-hi \
    --image-size 1024 --n-crops 6 --crop-size 320 --limit 14000
```
First samples the corpus to calibrate the Q\* references (`quality_reference.json`),
then runs A1 + A2 + the lesion/anatomy priors once and writes memmaps to
`data/cache/`: `images.npy` (1024 px retinal fields), `lesions.npy`,
`anatomy.npy`, `crops.npy` (per-image local-window centres) and `meta.csv`.
Builds patient-level splits and **asserts no patient crosses them**. Prints the
Q\* routing distribution, per-axis means, and lesion-visibility by DR grade.

⚠️ **Disk.** A 1024 px cache costs ~3.2 MB per image — all 35,126 images need
~110 GB. The script refuses to start if the cache would not fit and tells you
what to lower. `--limit N` takes a stratified, patient-grouped subset.

### Step 2b — pretrain the lesion experts on real masks (~40 min)
```bash
.venv/bin/python scripts/10_lesion_pretrain.py --image-size 448
```
Trains the A3 MoE against ophthalmologist annotations and reports per-channel
Dice **versus human masks**. Writes `outputs/lesion_pretrain/lesion_encoder.pt`.
Channels the source datasets do not annotate (NV, ME) are masked out of the loss
rather than scored as all-negative.

### Step 3 — staged training
```bash
.venv/bin/python scripts/03_train.py --stage-epochs 5,12,14,14 \
    --samples-per-epoch 6000 --max-val 1500 --batch-size 4 --grad-accum 4 \
    --global-size 448 --crop-input 224 --n-crops 6 \
    --lesion-encoder outputs/lesion_pretrain/lesion_encoder.pt
```
Loads real RETFound weights (hard-fails if it cannot), transfers the
lesion-pretrained experts, allocates GLA-LoRA ranks on a real batch, then runs
the four stages. After every epoch the per-sample losses feed the hard-example
miner and the sampler is rebuilt. Saves `best.pt` on
`0.5·QWK + 0.3·MacroF1 + 0.2·minority recall`, fits the temperature calibrator,
and writes `gla_lora.json`, `history.json`, `hardness.npy`.

Useful flags: `--time-budget-min N` stops cleanly at a wall-clock limit;
`--allow-no-retfound` is the ablation escape hatch; `--stage-epochs 1,1,1,1` is
a fast smoke test of all four stages.

### Step 4 — full evaluation (~12 min)
```bash
.venv/bin/python scripts/04_evaluate.py --external-cache data/cache_external
```
All 38 parameters across 7 groups, plus confusion matrix, decision-gate
distribution and a corruption sweep on real retinas. Writes `evaluation.json`.
Prognosis prints `NOT EVALUATED` rather than inventing numbers.

### Step 5 — single-image clinical report (~30 s)
```bash
.venv/bin/python scripts/05_predict.py --image <any_fundus.png>
```
Runs A1→A10 end to end on one photograph — including the local crop extraction,
which mirrors the training dataloader exactly — and prints the report: triage
decision, the five Q\* axes, graded probability bars, lesion table, uncertainty,
counterfactual ΔP, and the **grounded explanation chain** (each cited lesion with
its anatomical region, attention share, causal Δgrade, and the overall
chain-confidence score). Saves a 4-panel XAI figure to `outputs/predictions/`.

### Step 6 — deployment optimisation (~15 min)
```bash
.venv/bin/python scripts/06_deploy.py --epochs 2 --max-train 6000
```
Distils the pipeline into an image-only MobileNetV3 student, then benchmarks
FP32 / FP16 / INT8-dynamic / ONNX for size and latency. Writes
`distilled_metrics.json`, `student.pt`, `student.onnx`.

### Step 7 — model comparison (~2 h, resumable)
```bash
.venv/bin/python scripts/07_compare_models.py --epochs 1 \
    --max-train 2500 --max-test 2000
```
16 baselines (CNNs, transformers, linear probe, uniform-LoRA control) plus 6
proposed variants and ablations, all on an identical budget that is recorded in
the JSON. Results append incrementally, so completed entries are skipped on
rerun — use `--only name1,name2` to pick a subset.

### Step 9 — screening triage (~8 min)
```bash
.venv/bin/python scripts/09_screening.py --external-cache data/cache_external
```
Selects referral thresholds on **validation** against sensitivity or NPV floors,
applies them unchanged to test and to the external cohort, and reports referral
rate, workload reduction and missed cases. Writes `screening.json`, which
step 05 then reads to make its triage decision.

### Everything at once
```bash
./run_all.sh          # full run
./run_all.sh quick    # 4,000-image subset for a smoke test
```

---

## Running on Windows

The code is platform-portable (all paths go through `pathlib`, no shell calls
except `curl`, which ships with Windows 10 build 17063+). The commands below are
the Windows equivalents of the steps above.

> **Not verified on Windows.** This project was developed and run end-to-end on
> macOS/Apple Silicon. The Windows path is portable by construction and by code
> inspection, but the numbers in this README were produced on macOS. The
> Windows-specific notes below are the known differences, not a test report.

### Setup (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked by execution policy, either run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the session, or
skip activation entirely and call `.\.venv\Scripts\python.exe` directly — every
command below works that way too.

**Install PyTorch first, matching your hardware:**

```powershell
# NVIDIA GPU (CUDA 12.1) - strongly preferred
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only - works, but training will take many hours
pip install torch torchvision

pip install -r requirements.txt
```

### Kaggle credentials

Place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`, or set the
environment variables for the session:

```powershell
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY      = "your_key"
```

### The steps

```powershell
python scripts\08_selftest.py
python scripts\01_download_data.py --dataset eyepacs-hi
python scripts\11_download_lesions.py
python -u scripts\02_preprocess.py --root data\raw\eyepacs-hi --image-size 1024 --n-crops 6 --crop-size 320 --limit 12000
python -u scripts\10_lesion_pretrain.py --image-size 224 --batch-size 16
python -u scripts\03_train.py --stage-epochs 5,12,14,14 --samples-per-epoch 6000 --max-val 1500 --batch-size 4 --grad-accum 4 --global-size 224 --crop-input 224 --n-crops 6 --lesion-encoder outputs\lesion_pretrain\lesion_encoder.pt
python -u scripts\04_evaluate.py --external-cache data\cache_external
python -u scripts\05_predict.py --image path\to\fundus.png
python -u scripts\06_deploy.py --epochs 2
python -u scripts\07_compare_models.py --epochs 1 --max-train 2500 --max-test 2000
python -u scripts\09_screening.py --external-cache data\cache_external
```

Or run everything with the batch launcher:

```powershell
.\run_all.bat          # full run
.\run_all.bat quick    # 4,000-image subset
```

### Windows-specific notes

- **`--workers` on step 02.** Windows uses `spawn` rather than `fork`, so each
  pool worker re-imports the module and pays a start-up cost. The
  `if __name__ == "__main__"` guard needed for this is already in place. Start
  with `--workers 4` and raise it if your CPU has headroom.
- **`num_workers` stays 0 for training.** The default already is 0. This is a
  macOS deadlock workaround, but it is also the right choice on Windows, where
  spawn overhead usually outweighs the ~2 ms/image loading cost.
- **CUDA is auto-detected.** `pick_device()` prefers `cuda` → `mps` → `cpu`, so
  an NVIDIA GPU is used automatically. With ≥12 GB VRAM, raise
  `--batch-size` to 32–64 and `--samples-per-epoch` to 24000 — the batch-16
  advice in this README is an 8 GB unified-memory constraint, not a general one.
- **Long paths.** The extracted dataset nests as
  `data\raw\eyepacs-224\colored_images\colored_images\<Grade>\<id>.png`. If you
  clone into an already-deep directory, enable long paths:
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`
- **Disk.** Budget ~10 GB: ~2 GB dataset plus a ~6.5 GB preprocessing cache.
- **INT8 quantisation** in step 06 selects `fbgemm` on x86 Windows automatically
  (the `qnnpack` branch is ARM-only), so that variant should succeed rather than
  fall back.

---

## Architecture → code map

| Block | Description | Implementation |
|---|---|---|
| **A1** | Adaptive Quality & Domain Gate — 5-axis Q\* routing | [`a1_quality.py`](src/dr/modules/a1_quality.py) |
| **A2** | Adaptive Lesion-Preserving Preprocessing — learned fusion | [`a2_preprocess.py`](src/dr/modules/a2_preprocess.py) |
| **A3** | Lesion Specialist MoE — adaptive gate, real-mask pretrained | [`a3_lesion_moe.py`](src/dr/modules/a3_lesion_moe.py) · [`lesion_datasets.py`](src/dr/data/lesion_datasets.py) |
| **A4** | Real RETFound + GLA-LoRA + staged fine-tuning | [`a4_backbone_lora.py`](src/dr/modules/a4_backbone_lora.py) |
| **A5** | Bidirectional Pathology↔Anatomy Fusion / Graph | [`a5_fusion_gnn.py`](src/dr/modules/a5_fusion_gnn.py) |
| **A6** | Hierarchical Ordinal + boundary + contrastive | [`a6_ordinal.py`](src/dr/modules/a6_ordinal.py) |
| **A7** | Temporal Prognostic Engine | [`a7_temporal.py`](src/dr/modules/a7_temporal.py) |
| **A8** | Multimodal Clinical Fusion | [`a8_multimodal.py`](src/dr/modules/a8_multimodal.py) |
| **A9** | Lesion-Grounded Explainability chain | [`a9_xai.py`](src/dr/modules/a9_xai.py) |
| **A10** | Uncertainty & Calibration + Decision Gate | [`a10_uncertainty.py`](src/dr/modules/a10_uncertainty.py) |
| Validation & Generalization | corruptions, cross-domain, subgroups | [`04_evaluate.py`](scripts/04_evaluate.py) |
| Deployment Optimization | distillation, FP16/INT8, ONNX | [`06_deploy.py`](scripts/06_deploy.py) |
| Model Comparison (20+) | CNNs, transformers, adaptation, ablations | [`07_compare_models.py`](scripts/07_compare_models.py) |
| Final Clinical AI Report | triage, grade, flags, lesions, XAI panel | [`report.py`](src/dr/report.py) + [`05_predict.py`](scripts/05_predict.py) |
| Screening triage | val-fitted operating points, NPV / workload | [`screening.py`](src/dr/screening.py) + [`09_screening.py`](scripts/09_screening.py) |
| Class imbalance & hard-example mining | n^-0.5 sampling, boundary/low-margin bonuses | [`sampling.py`](src/dr/sampling.py) |
| Real lesion pretraining | IDRiD / DDR / FGADR → transfer to grading | [`10_lesion_pretrain.py`](scripts/10_lesion_pretrain.py) · [`11_download_lesions.py`](scripts/11_download_lesions.py) |

### Design points worth knowing

**GLA-LoRA rank allocation is measured, not guessed.** Before any adapter
exists, `gla_importance` measures three things per ViT block on a real batch:
the gradient norm the task induces there, the correlation between patch-token
energy and lesion density, and the share of that block's attention landing on
lesion patches. Ranks follow the pooled score, with lesion-sensitive blocks
pinned to `r_max`. One correlation is a noisy estimator; three signals that
disagree are informative, and the allocation is written to `gla_lora.json` for
every run so it can be inspected rather than assumed.

**The local branch is load-bearing, and the self-test proves it.** `08_selftest.py`
asserts that removing the crops measurably changes the predicted grade — a local
branch that is merely wired up but ignored would pass a shape check and fail
this one. It likewise asserts that the MoE gate responds to Q\* and hard-example
context, that Q\* falls when an image is degraded, that the ALPP gate produces
different weights for different images, and that a missing RETFound checkpoint
raises instead of falling through.

**Q\* references are calibrated per corpus, not hard-coded.** Lesion CNR and
Laplacian variance are not comparable across resolutions or denoising pipelines
— a downsampled, gaussian-filtered mirror of EyePACS has a far lower noise floor
than raw camera output, so the same retina scores several times higher.
`02_preprocess.py` samples the corpus, maps its 5th/85th CNR percentiles onto
[0,1], and writes `quality_reference.json`. Without this the lesion-visibility
axis pins at 1.0 for every image and contributes nothing. Pass
`--quality-reference` when caching an external set, so its `Q_domain` is
measured against the *training* domain rather than against itself.

**A6 monotonicity is structural.** Thresholds are `b_0 − cumsum(softplus(δ))`,
so `P(Y>0) ≥ P(Y>1) ≥ P(Y>2) ≥ P(Y>3)` cannot be violated and the derived
categorical distribution is always non-negative.

**Splits are patient-level.** Each EyePACS patient has a left and a right eye
with strongly correlated grades; a per-image split leaks the same patient into
train and test. `split_by_patient` groups on patient id and stratifies on the
patient's worst eye, and `02_preprocess.py` asserts no patient crosses splits.

**A9's training-time attribution is differentiable.** The consistency loss uses
the A5 cross-attention map rather than Grad-CAM, avoiding double backprop.
Grad-CAM and attention rollout are still used at report time. Only the leading
grid-aligned query tokens are reshaped back onto the image — the per-crop tokens
have no spatial position and would corrupt the map.

**λ_XAI starts at zero on purpose.** Attribution-consistency is a strong prior,
and applying it to a model whose attention has not yet organised teaches it to
match noise. The schedule holds λ at 0 through the frozen and adapter stages,
then ramps 0.02 → 0.05 as the encoder settles.

---

## What is real, and what is not

This matters more than any accuracy number, so it is stated plainly.

### Real
- **Images and DR grades** — all 35,126 EyePACS fundus photographs, graded 0–4 by
  licensed clinicians. Grade distribution 73.5 / 7.0 / 15.1 / 2.5 / 2.0 %.
- **All diagnostic, ordinal, calibration, robustness, subgroup and deployment
  metrics** — measured on a patient-disjoint test split.
- **A1 quality scores and camera-domain fingerprints** — computed from the pixels.
- **Latency, model size, compression** — measured on this machine.

### Lesion supervision — now real, with one caveat
- **The A3 experts are pretrained on real ophthalmologist masks**: IDRiD (81
  images) and DDR (757), 838 in total, covering microaneurysms, haemorrhages,
  hard exudates and soft exudates. `scripts/10_lesion_pretrain.py` trains them
  and reports per-channel Dice **against human annotation**, so that number
  means what it says.
- **NV and ME are still not real.** Neither set annotates neovascularisation or
  macular oedema, so those two experts are masked out of the pretraining loss
  and fall back to the morphological priors in
  [`lesion_priors.py`](src/dr/data/lesion_priors.py) during DR training. FGADR
  does annotate NV/IRMA but needs a signed request form; drop it at
  `data/raw/lesion/fgadr/` and the loader picks it up automatically.
- **EyePACS itself still ships no pixel masks.** The A9 attribution-consistency
  loss and its Dice metric are computed against the morphological priors on
  EyePACS images, so *those specific numbers* remain "agreement with a prior",
  not with a grader. The grounded-explanation object carries a real
  `annotated_dice` field, populated only where human masks exist.

### Not evaluated
- **A7 prognosis (4 metrics).** EyePACS is cross-sectional — one visit per
  patient, no follow-up grades. The temporal engine is implemented and
  unit-tested, but there is nothing in this dataset to train or score it on, so
  `04_evaluate.py` prints `NOT EVALUATED` instead of inventing numbers. Supply a
  longitudinal cohort to activate it.
- **A8 clinical fusion** runs with its gate hard-zeroed on EyePACS, which has no
  clinical metadata. Pass `--clinical-csv` to activate it.

### RETFound weights
The runs here load **real RETFound MAE ViT-Large/16 CFP weights** (the published
Nature checkpoint, via the ungated `bitfount/RETFound_MAE` mirror), verified at
100% backbone tensor coverage with zero missing and zero unexpected keys. The
loader raises `RETFoundCheckpointError` rather than falling back to ImageNet, so
a run either used RETFound or did not start.

Note on naming: there is no public checkpoint called "RETFound **Plus**". The
RETFound family on HuggingFace is the original `RETFound_mae_natureCFP` plus the
later `RETFound_mae_meh` / `_shanghai` / `RETFound_dinov2_*` variants, all of
which are gated behind a licence acceptance. Point `--retfound-ckpt` at any of
them (`YukunZhou/RETFound_mae_meh:RETFound_mae_meh.pth`) once you have accepted
the terms; the loader handles the key remapping and positional-embedding resize
either way.

---

## Model comparison and ablations

14 entries at an identical budget (1 epoch, 2,500 train / 2,000 test, seed 1337).
Full table in `outputs/comparison/comparison.json`.

| model | family | acc | QWK | refAUC | trainable |
|---|---|---|---|---|---|
| regnety_016 | CNN | 0.7560 | **0.4743** | 0.7920 | 10.3 M |
| abl_no_allora | Proposed | 0.7370 | 0.4732 | 0.7860 | 11.0 M |
| abl_no_xai | Proposed | 0.7225 | 0.4654 | 0.7861 | 10.9 M |
| abl_no_fusion | Proposed | 0.7420 | 0.4516 | 0.7790 | 10.9 M |
| **complete** | Proposed | 0.7335 | 0.4365 | 0.7777 | 10.9 M |
| resnext50 | CNN | 0.7095 | 0.4295 | 0.7463 | 23.0 M |
| efficientnet_b0 | CNN | 0.6870 | 0.4140 | 0.7592 | 4.0 M |
| densenet121 | CNN | 0.7170 | 0.3913 | 0.7103 | 7.0 M |
| mobilenetv3_large | CNN | 0.7235 | 0.3102 | 0.7344 | 4.2 M |
| resnet18 | CNN | 0.7105 | 0.2494 | 0.6750 | 11.2 M |
| abl_no_ordinal | Proposed | 0.3545 | 0.1460 | 0.7762 | 10.9 M |
| resnet50 | CNN | 0.7375 | 0.0245 | 0.6711 | 23.5 M |
| convnext_tiny | CNN | 0.7400 | 0.0000 | 0.5255 | 27.8 M |
| abl_no_moe | Proposed | 0.7400 | 0.0000 | 0.5266 | 10.9 M |

**Read this as a component screen, not a verdict on converged performance** — the
actual trained model reached QWK 0.5944 on the full test split, above every row
here. Single seed, so QWK gaps under ~0.05 are noise.

**Load-bearing components:**
- **A6 ordinal head** — removing it costs QWK 0.4365 → 0.1460, the largest effect
  measured.
- **A3 lesion MoE** — removing it collapses the model to QWK 0.0000; its
  auxiliary loss acts as a training stabiliser.

**Components that do not currently earn their place** (removing each *raises* QWK):
- **A4 adaptive rank allocation** — uniform rank 9 scored 0.4732 vs 0.4365.
- **A9 consistency loss** — 0.4654; the term never moved off 0.93 during training.
- **A5 lesion→anatomy edges** — 0.4516.

All three are driven by the weak morphological lesion priors, which do not track
DR grade at 224 px (see the honesty section). They cannot be judged fairly until
run with real lesion masks or the 1024 px imagery. `convnext_tiny` and
`resnet50` collapsing is a budget artefact — both need a lower LR than the
shared 3e-4.

---

## Output artefacts

After a full run, `outputs/retfound_plus_laft_xai/` contains:

| file | contents |
|---|---|
| `best.pt` | checkpoint (weights + config + AL-LoRA ranks), selected on val QWK |
| `gla_lora.json` | the three per-block importance signals (G/L/A), the pooled S_l, and the allocated ranks |
| `hardness.npy` | final per-sample hard-example score, indexed by cache row |
| `quality_reference.json` | corpus-calibrated Q* references written by step 02 |
| `lesion_pretrain/lesion_encoder.pt` | A3 experts pretrained on IDRiD + DDR, with per-channel Dice vs human masks |
| `history.json` | per-epoch loss and validation metrics |
| `calibrator.pt` | temperature scaler fitted on validation |
| `evaluation.json` | all 38 metrics, confusion matrix, subgroup breakdowns |
| `screening.json` | triage operating points for val / test / external |
| `distilled_metrics.json` | teacher vs student, quantisation and ONNX benchmarks |
| `student.pt`, `student.onnx` | distilled edge model |

`outputs/predictions/` holds per-image reports (`.txt`) and XAI panels (`.png`);
`outputs/comparison/comparison.json` holds the model-comparison leaderboard.

---

## Hardware notes

Developed and run on an **Apple M3 (8 GB, MPS)**. Measured behaviour on that box:

- **Batch 16 is the throughput sweet spot** (~11 img/s). Batch 32 thrashes
  unified memory and runs ~10× slower *per image* — not an OOM, just swap.
- **`num_workers=0` is the default.** Forked DataLoader workers deadlock against
  the open memmap on macOS, and loading costs only ~2 ms/image against ~80 ms of
  compute, so workers buy nothing here. Raise it on Linux/CUDA.
- Throughput oscillates as the 5.3 GB image memmap is paged in and out; one
  epoch varied between 17 and 33 minutes for this reason.

On a CUDA GPU use `--backbone vit_large_patch16_224 --retfound-ckpt ...` with a
larger batch for the intended full-scale configuration.

The images in `eyepacs-224` are pre-resized to 224 px, which is why Mild DR is
undetectable (microaneurysms are ~1 px); `--dataset eyepacs-hi` fetches the
~1024 px mirror (7.8 GB) so the preprocessing chain works at native lesion scale.

---

## Known issues fixed during development

Recorded because they are easy to hit again:

- **MPS `min`/`max` backward emits scatter index −1** on the flattened
  attribution map. Fixed by detaching the min-max bounds in `a9_xai._minmax` —
  which is also the correct semantics, since gradient should not flow through
  *which* pixel was extremal.
- **`timm.num_features` is not always the pooled output width** (mobilenetv3
  reports 576 but emits 1024). Both `06_deploy.py` and `07_compare_models.py`
  now probe it with a dummy forward.
- **Stacking a balanced sampler with class-balanced loss weights** double-corrects
  the imbalance and drives short runs to predict only rare classes (resnet18 hit
  7% accuracy). `07_compare_models.py` defaults to `--cb-beta 0.0` so the sampler
  alone does the rebalancing.
- **INT8 dynamic quantisation needs the `qnnpack` engine** on Apple Silicon;
  `fbgemm` is x86-only. Note it only touches `nn.Linear`, so a mostly-conv
  student barely shrinks — use static/QAT for real gains.
- **`torch.onnx` writes weights to a sibling `.onnx.data` file.** Reporting the
  graph size alone understates the deployed footprint by ~20×.
