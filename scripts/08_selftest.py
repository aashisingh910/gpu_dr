#!/usr/bin/env python3
"""Step 08 - self-test every block (A1-A10) on synthetic tensors.

Runs in seconds with no dataset, so architecture bugs surface before a long
training run. Also exercises the A7 temporal path, which real EyePACS cannot
train (no follow-up visits).

    python scripts/08_selftest.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import default_config  # noqa: E402
from dr.data.lesion_priors import LesionPriorExtractor  # noqa: E402
from dr.model import build_model  # noqa: E402
from dr.modules import a1_quality, a2_preprocess  # noqa: E402
from dr.modules.a3_lesion_moe import lesion_supervision_loss  # noqa: E402
from dr.modules.a5_fusion_gnn import pathology_anatomy_consistency  # noqa: E402
from dr.modules.a6_ordinal import (boundary_loss,  # noqa: E402
                                   class_balanced_focal_loss, coral_targets,
                                   hierarchical_loss, ordinal_contrastive_loss,
                                   ordinal_loss)
from dr.sampling import (HardExampleState, sampling_weights,  # noqa: E402
                         update_hardness)
from dr.modules.a7_temporal import (TemporalPrognosticEngine,  # noqa: E402
                                    progression_loss)
from dr.modules.a9_xai import (attribution_consistency_loss,  # noqa: E402
                               counterfactual_image, dice_score, explain,
                               explanation_report, pointing_game,
                               xai_weight_for_stage)
from dr.modules.a10_uncertainty import (build_calibrator, decision_gate,  # noqa: E402
                                        mc_predict)

PASS, FAIL = "  PASS", "  FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        msg = fn() or ""
        results.append((name, True, msg))
        print(f"{PASS}  {name} {msg}")
    except Exception as e:  # noqa: BLE001
        results.append((name, False, str(e)))
        print(f"{FAIL}  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


def synthetic_fundus(size: int = 512) -> np.ndarray:
    """A crude but structurally realistic fundus: circular FOV, disc, vessels,
    bright and dark lesions."""
    rng = np.random.default_rng(0)
    img = np.zeros((size, size, 3), np.uint8)
    c, r = size // 2, int(size * 0.45)
    yy, xx = np.mgrid[:size, :size]
    fov = ((yy - c) ** 2 + (xx - c) ** 2) <= r * r
    img[fov] = (40, 90, 180)                       # BGR retinal background
    import cv2
    cv2.circle(img, (int(c + r * 0.55), c), int(size * 0.05), (200, 235, 250), -1)
    for a in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        p = (int(c + r * 0.55), c)
        q = (int(p[0] + np.cos(a) * r * 0.8), int(p[1] + np.sin(a) * r * 0.8))
        cv2.line(img, p, q, (25, 45, 120), 2)
    for _ in range(25):                            # dark lesions
        x, y = rng.integers(c - r // 2, c + r // 2, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 5)), (20, 35, 110), -1)
    for _ in range(15):                            # bright exudates
        x, y = rng.integers(c - r // 2, c + r // 2, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 7)), (120, 230, 245), -1)
    img[~fov] = 0
    return img


def main() -> None:
    cfg = default_config()
    cfg.backbone.name = "vit_tiny_patch16_224"
    cfg.backbone.pretrained = False               # offline-safe self-test
    cfg.backbone.retfound_ckpt = None             # the checkpoint gate is tested
    cfg.backbone.require_retfound = False         # separately, below
    cfg.fusion.fused_dim = 192
    cfg.preproc.cache_size = 512
    cfg.preproc.global_size = 224
    cfg.preproc.crop_size = 160
    cfg.preproc.crop_input = 112
    cfg.preproc.n_crops = 4
    cfg.moe.mask_size = 56
    device = torch.device("cpu")
    B, S, M, C = 2, cfg.preproc.global_size, cfg.moe.mask_size, cfg.preproc.n_crops
    img = synthetic_fundus()

    print("\n=== A1  Adaptive Quality & Domain Gate ===")
    rep = {}

    def t_a1():
        r = a1_quality.assess(img, cfg.quality)
        rep["r"] = r
        assert 0.0 <= r.score <= 1.0
        assert r.decision in ("retake", "enhance", "accept")
        assert r.fov_mask.sum() > 1000
        ax = r.axes()
        assert set(ax) == {"quality", "domain", "lesion", "blur", "illumination"}
        assert all(0.0 <= v <= 1.0 for v in ax.values()), ax
        return (f"Q*={r.score:.3f} (legacy {r.legacy_score:.3f}) "
                f"decision={r.decision} domain={r.domain} "
                + " ".join(f"{k[:4]}={v:.2f}" for k, v in ax.items()))
    check("A1 upgraded 5-axis quality gate", t_a1)

    def t_a1_routing():
        """Q* must actually route: a blurred copy has to fall a routing tier."""
        import cv2
        blurred = cv2.GaussianBlur(img, (0, 0), 6.0)
        rb = a1_quality.assess(blurred, cfg.quality)
        assert rb.score < rep["r"].score, (rb.score, rep["r"].score)
        assert rb.q_lesion <= rep["r"].q_lesion + 1e-6
        return (f"sharp Q*={rep['r'].score:.3f}/{rep['r'].decision} -> "
                f"blurred Q*={rb.score:.3f}/{rb.decision} "
                f"(Q_lesion {rep['r'].q_lesion:.2f} -> {rb.q_lesion:.2f})")
    check("A1 Q* routing responds to degradation", t_a1_routing)

    print("\n=== A2  Adaptive Lesion-Preserving Preprocessing ===")
    pre = {}

    def t_a2():
        p = a2_preprocess.run(img, rep["r"], cfg.preproc)
        pre["p"] = p
        CS = cfg.preproc.cache_size
        assert p.image.shape == (CS, CS, 3)
        assert p.fov_mask.shape == (CS, CS)
        assert 0.0 <= p.vessels.max() <= 1.0
        return f"I* {p.image.shape} vessels max={p.vessels.max():.2f}"
    check("A2 cache-time preprocessing", t_a2)

    def t_alpp():
        """The learned fusion must produce per-image, not constant, weights."""
        from dr.modules.a2_preprocess import ALPP
        alpp = ALPP(cfg.preproc)
        torch.manual_seed(0)
        with torch.no_grad():
            for lin in alpp.head.modules():
                if isinstance(lin, torch.nn.Linear):
                    lin.weight.normal_(0, 0.5)     # untrained head is zero-init
        a = torch.rand(1, 3, 224, 224) * 0.3
        b = torch.rand(1, 3, 224, 224) * 0.3 + 0.6
        oa, wa = alpp(a, torch.rand(1, 5))
        ob, wb = alpp(b, torch.rand(1, 5))
        assert oa.shape == a.shape and 0.0 <= float(oa.min()) and float(oa.max()) <= 1.0
        assert torch.allclose(wa.sum(-1), torch.ones(1), atol=1e-5)
        assert float((wa - wb).abs().max()) > 1e-3, "gate is image-independent"
        return (f"w(dark)={np.round(wa[0].detach().numpy(), 3).tolist()} "
                f"w(bright)={np.round(wb[0].detach().numpy(), 3).tolist()}")
    check("A2 ALPP learned fusion weights", t_alpp)

    print("\n=== Weak lesion priors + anatomy ===")
    pri = {}

    def t_priors():
        ex = LesionPriorExtractor(out_size=M)
        pr = ex(pre["p"].image, pre["p"].fov_mask)
        pri["p"] = pr
        assert pr.masks.shape == (6, M, M)
        assert pr.anatomy.shape == (5, M, M)
        return (f"burden={np.round(pr.burden(), 4).tolist()} "
                f"disc={pr.disc_center} macula={pr.macula_center}")
    check("lesion priors", t_priors)

    def t_crops():
        from dr.data.lesion_priors import generate_lesion_crops
        pr = pri["p"]
        c = generate_lesion_crops(
            pr.masks, pr.anatomy, pre["p"].fov_mask, n_crops=C,
            crop_frac=cfg.preproc.crop_size / cfg.preproc.cache_size,
            disc_center=pr.disc_center, macula_center=pr.macula_center)
        assert c.shape == (C, 2), c.shape
        # centres are returned in fov_mask's pixel grid (cache_size x cache_size),
        # not normalised 0-1 - see generate_lesion_crops' own docstring and how
        # scripts/02_preprocess.py/dr.data.eyepacs.CachedEyePACS consume crops.npy
        # directly as pixel-space window centres.
        CS = cfg.preproc.cache_size
        assert c.min() >= 0.0 and c.max() <= CS - 1, (c.min(), c.max(), CS)
        uniq = len({tuple(np.round(v, 3)) for v in c})
        assert uniq >= 2, "crop generator collapsed onto one point"
        return f"{C} crop centres, {uniq} distinct: {np.round(c, 2).tolist()}"
    check("local lesion crop generator", t_crops)

    print("\n=== A3-A6, A8, A9  full model forward ===")
    model = build_model(cfg).to(device)
    x = torch.rand(B, 3, S, S)
    crops = torch.rand(B, C, 3, cfg.preproc.crop_input, cfg.preproc.crop_input)
    anat = torch.rand(B, 5, M, M)
    masks = (torch.rand(B, 6, M, M) > 0.9).float()
    qax = torch.rand(B, 5)
    hard_t = torch.rand(B)
    y = torch.tensor([0, 3])
    batch = {"image": x, "crops": crops, "anatomy": anat,
             "quality_axes": qax, "hardness": hard_t}

    def t_retfound_gate():
        """require_retfound must hard-fail rather than fall back to ImageNet."""
        from dr.modules.a4_backbone_lora import (RETFoundCheckpointError,
                                                 RETFoundPlusBackbone)
        import copy
        bad = copy.deepcopy(cfg.backbone)
        bad.require_retfound = True
        bad.retfound_ckpt = "/nonexistent/retfound.pth"
        try:
            RETFoundPlusBackbone(bad, img_size=S)
        except RETFoundCheckpointError as e:
            return f"correctly refused a missing checkpoint: {str(e)[:70]}..."
        raise AssertionError("a missing RETFound checkpoint was silently tolerated")
    check("A4 RETFound checkpoint assertion", t_retfound_gate)

    def t_adapters():
        info = model.fit_adapters(x, masks)
        nb = len(model.backbone.vit.blocks)
        assert len(info["ranks"]) == nb
        assert all(cfg.backbone.lora_rank_min <= r <= cfg.backbone.lora_rank_max
                   for r in info["ranks"])
        for k in ("grad_G", "lesion_L", "attn_A", "importance_S"):
            assert len(info[k]) == nb, k
            assert all(0.0 <= v <= 1.0 for v in info[k]), (k, info[k])
        assert len(set(info["ranks"])) > 1, "GLA-LoRA gave every layer one rank"
        return (f"ranks={info['ranks']} ({info['n_at_max_rank']} at r_max) "
                f"params={info['lora_params']:,}")
    check("A4 GLA-LoRA gradient/lesion/attention rank allocation", t_adapters)

    def t_stages():
        seen = []
        for st in cfg.train.stages:
            g = model.set_stage(st)
            seen.append((st.name, g["backbone"]["unfrozen_blocks"],
                         g["n_lora_params"] > 0))
            groups = model.optimizer_param_groups(st, 0.02)
            names = [gg["name"] for gg in groups]
            assert "head" in names
            if st.train_lora:
                assert "lora" in names, st.name
            if st.unfreeze_frac > 0:
                assert "backbone" in names, st.name
        assert seen[0][1] == 0 and seen[-1][1] == len(model.backbone.vit.blocks)
        assert [s[1] for s in seen] == sorted(s[1] for s in seen), "not monotone"
        model.set_stage(cfg.train.stages[1])
        return " -> ".join(f"{n}:{u}blk{'+lora' if l else ''}" for n, u, l in seen)
    check("A4 staged fine-tuning schedule", t_stages)

    out = {}

    def t_forward():
        o = model(batch)
        out["o"] = o
        assert o["ordinal"].class_probs.shape == (B, 5)
        p = o["ordinal"].class_probs
        assert torch.allclose(p.sum(1), torch.ones(B), atol=1e-4)
        cum = o["ordinal"].cum_probs
        assert (cum[:, :-1] >= cum[:, 1:] - 1e-6).all(), "ordinal monotonicity violated"
        assert o["moe"].evidence.shape == (B, 6, M, M)
        assert o["attribution"].shape == (B, 1, M, M)
        assert o["ordinal"].hier_logits.shape == (B, len(cfg.head.hierarchy))
        assert o["ordinal"].projection.shape == (B, cfg.head.proj_dim)
        assert o["local_cls"] is not None and o["local_cls"].shape[1] == C
        return (f"probs {tuple(p.shape)} monotone-OK, {C} local crops encoded, "
                f"alphas={np.round(o['moe'].alphas[0].detach().numpy(), 3).tolist()}")
    check("A3/A5/A6/A8/A9 forward (global + local)", t_forward)

    def t_local_matters():
        """The local branch must change the prediction, or it is decorative."""
        with torch.no_grad():
            a = model({k: v for k, v in batch.items() if k != "crops"})
            b = model(batch)
        d = float((a["ordinal"].expected_grade - b["ordinal"].expected_grade).abs().max())
        assert d > 1e-5, "local crops had no effect on the output"
        return f"max |dgrade| between global-only and global+local = {d:.4f}"
    check("local lesion branch influences the grade", t_local_matters)

    def t_bidirectional():
        f = out["o"]["fusion"]
        assert f.attn_p2a.shape[-1] == len(cfg.fusion.anatomy_regions)
        assert f.attn_a2p.shape[-2] == len(cfg.fusion.anatomy_regions)
        assert f.probe_p.shape == f.probe_a.shape == (B, cfg.fusion.probe_dim)
        lpa = pathology_anatomy_consistency(f)
        assert torch.isfinite(lpa) and float(lpa) >= 0
        return (f"P->A {tuple(f.attn_p2a.shape)}  A->P {tuple(f.attn_a2p.shape)}  "
                f"L_PA={float(lpa):.4f}")
    check("A5 bidirectional pathology<->anatomy fusion", t_bidirectional)

    def t_moe_gate():
        """The adaptive gate must respond to its context, not just the pixels."""
        with torch.no_grad():
            lo = model({**batch, "quality_axes": torch.zeros(B, 5),
                        "hardness": torch.zeros(B)})
            hi = model({**batch, "quality_axes": torch.ones(B, 5),
                        "hardness": torch.ones(B) * 3})
        d = float((lo["moe"].alphas - hi["moe"].alphas).abs().max())
        assert d > 1e-6, "MoE gate ignores Q and hard-example context"
        return f"max |dalpha| across quality/hardness context = {d:.5f}"
    check("A3 adaptive MoE gate uses (Z_G, Z_L, Q, H)", t_moe_gate)

    def t_freeze():
        pr = model.param_report()
        assert pr["trainable_params"] < pr["total_params"]
        vit_grad = [p.requires_grad for n, p in model.backbone.vit.named_parameters()
                    if "A" not in n.split(".")[-1] and "B" not in n.split(".")[-1]]
        return (f"trainable {pr['trainable_params']:,}/{pr['total_params']:,} "
                f"({pr['trainable_pct']}%)")
    check("A4 backbone frozen", t_freeze)

    def t_losses():
        o = out["o"]
        lo = cfg.loss
        counts = torch.tensor([1000.0, 100.0, 200.0, 50.0, 30.0])
        terms = {
            "ord": (lo.w_ordinal, ordinal_loss(o["ordinal"], y)),
            "hier": (lo.w_hierarchical,
                     hierarchical_loss(o["ordinal"], y, cfg.head.hierarchy_thresholds)),
            "bnd": (lo.w_boundary, boundary_loss(o["ordinal"], y, lo.boundary_gamma)),
            "con": (lo.w_contrastive,
                    ordinal_contrastive_loss(o["ordinal"].projection, y,
                                             lo.contrastive_temp)),
            "les": (lo.w_lesion,
                    lesion_supervision_loss(o["moe"], masks,
                                            (masks.amax((2, 3)) > 0.5).float())),
            "pa": (lo.w_pa, pathology_anatomy_consistency(o["fusion"])),
            "cbf": (lo.w_cbf, class_balanced_focal_loss(
                o["ordinal"].logits_ce, y, counts,
                weight_power=cfg.sampling.class_weight_power)),
            "xai": (0.05, attribution_consistency_loss(o["attribution"], masks)),
        }
        for k, (_, v) in terms.items():
            assert torch.isfinite(v), f"{k} is not finite"
        total = sum(w * v for w, v in terms.values())
        total.backward()
        g = [p.grad.abs().sum().item() for p in model.parameters()
             if p.requires_grad and p.grad is not None]
        assert len(g) > 0 and sum(g) > 0, "no gradient reached trainable params"
        return (" ".join(f"{k}={float(v):.3f}" for k, (_, v) in terms.items())
                + f"; {len(g)} tensors got gradient")
    check("full objective (ord+hier+bnd+con+les+PA+XAI) backprops", t_losses)

    def t_coral():
        t = coral_targets(torch.tensor([0, 2, 4]), 5)
        expect = torch.tensor([[0., 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
        assert torch.equal(t, expect), t
        return "rank targets correct"
    check("A6 CORAL target encoding", t_coral)

    print("\n=== Sampling: moderate re-balancing + hard-example mining ===")

    def t_mining():
        n = 400
        rng = np.random.default_rng(0)
        labels = rng.choice(5, n, p=[0.73, 0.07, 0.15, 0.03, 0.02])
        idx = np.arange(n)
        st = HardExampleState.empty(n)
        w0 = sampling_weights(labels, idx, st, cfg.sampling)
        # class prior alone: rarer classes must be up-weighted, but not to the
        # degree full inverse-frequency would
        by_cls = {g: float(w0[labels == g].mean()) for g in range(5) if (labels == g).any()}
        assert by_cls[4] > by_cls[0], by_cls
        ratio = by_cls[4] / by_cls[0]
        counts = np.bincount(labels, minlength=5)
        expected = (counts[0] / counts[4]) ** 0.5
        assert abs(ratio - expected) / expected < 0.05, (ratio, expected)

        # now make grade-1/2 boundary errors expensive and check they rise
        losses = np.ones(n); preds = labels.copy(); targets = labels.copy()
        hardmask = np.isin(labels, [1, 2])
        losses[hardmask] = 5.0
        preds[hardmask] = labels[hardmask] + 1          # adjacent-class error
        margins = np.where(hardmask, 0.02, 0.9)
        update_hardness(st, idx, losses, preds, targets, margins, cfg.sampling)
        w1 = sampling_weights(labels, idx, st, cfg.sampling)
        lift = float(w1[hardmask].mean() / w0[hardmask].mean())
        assert lift > 1.3, lift
        return (f"n^-0.5 gives grade4/grade0 weight ratio {ratio:.2f} "
                f"(exact {expected:.2f}); mining lifts Mild/Moderate "
                f"boundary errors {lift:.2f}x")
    check("moderate sampling + hard-example mining", t_mining)

    def t_xai_schedule():
        sched = [round(xai_weight_for_stage(cfg.xai, i, 1.0), 4)
                 for i in range(len(cfg.train.stages))]
        assert sched[0] == 0.0, "lambda_XAI must start at zero"
        assert sched == sorted(sched), "lambda_XAI must be non-decreasing"
        assert sched[-1] > 0, "lambda_XAI never turns on"
        return f"lambda_XAI per stage = {sched}"
    check("A9 lambda_XAI 0 -> 0.02 -> 0.05 schedule", t_xai_schedule)

    def t_grounded():
        exp = explain(model, batch, index=0, topk=cfg.xai.grounding_topk)
        assert 0 <= exp.predicted_grade < 5
        assert 0.0 <= exp.chain_confidence <= 1.0
        for c in exp.citations:
            assert c.lesion in ("MA", "HE", "EX_H", "EX_S", "NV", "ME")
            assert c.anatomy in tuple(cfg.fusion.anatomy_regions) + ("none",)
        txt = explanation_report(exp)
        assert "Predicted grade" in txt
        return (f"grade {exp.predicted_grade}, {len(exp.citations)} grounded "
                f"citations, attention-on-lesion {exp.attention_on_lesion:.1%}, "
                f"chain confidence {exp.chain_confidence:.3f}")
    check("A9 lesion-grounded explanation chain", t_grounded)

    def t_lesion_data():
        from dr.config import RAW_DIR
        from dr.data.lesion_datasets import discover_all
        recs, summary = discover_all(RAW_DIR / "lesion")
        found = {k: v["images"] for k, v in summary.items()}
        if not recs:
            return ("no real lesion annotations present yet "
                    "(run the IDRiD/DDR fetch); MoE falls back to weak priors")
        chans = sorted({c for r in recs for c in r.masks})
        return f"{len(recs)} annotated images {found}, channels {chans}"
    check("real lesion annotation discovery", t_lesion_data)

    print("\n=== A7  Temporal Prognostic Engine ===")

    def t_temporal():
        d = cfg.fusion.fused_dim
        eng = TemporalPrognosticEngine(d, cfg.head.horizons)
        T = 3
        feats = torch.randn(B, T, d)
        times = torch.tensor([[0.0, 12.0, 24.0], [0.0, 6.0, 18.0]])
        pad = torch.tensor([[False, False, False], [False, False, True]])
        o = eng(feats, times, pad)
        assert o.risk_logits.shape == (B, 4)
        assert (o.survival <= 1.0).all() and (o.survival >= 0).all()
        assert (o.survival[:, :-1] >= o.survival[:, 1:] - 1e-6).all(), \
            "survival must be non-increasing"
        loss = progression_loss(o, torch.tensor([[0., 0, 1, 1], [0., 1, 1, 1]]),
                                torch.tensor([2, 1]), torch.tensor([1.0, 1.0]))
        loss.backward()
        return f"risk {tuple(o.risk_logits.shape)} survival monotone-OK loss={float(loss):.3f}"
    check("A7 temporal engine", t_temporal)

    print("\n=== A9  XAI: counterfactual + faithfulness metrics ===")

    def t_xai():
        m = pri["p"].masks.max(0)
        cf = counterfactual_image(pre["p"].image, m)
        assert cf.shape == pre["p"].image.shape
        changed = float(np.abs(cf.astype(int) - pre["p"].image.astype(int)).mean())
        attr = out["o"]["attribution"][0, 0].detach().numpy()
        d = dice_score(attr, m)
        pg = pointing_game(attr, m)
        return f"counterfactual mean|delta|={changed:.2f} dice={d:.3f} pointing={pg}"
    check("A9 counterfactual + metrics", t_xai)

    print("\n=== A10  Uncertainty, calibration, decision gate ===")

    def t_mc():
        u = mc_predict(model, {"image": x, "anatomy": anat}, n_samples=4)
        assert u.probs.shape == (B, 5)
        assert (u.entropy >= 0).all() and (u.entropy <= 1.001).all()
        return (f"H={np.round(u.entropy.numpy(), 3).tolist()} "
                f"epistemic={np.round(u.epistemic.numpy(), 3).tolist()}")
    check("A10 MC-dropout ensemble", t_mc)

    def t_calib():
        rng = np.random.default_rng(0)
        logits = torch.as_tensor(rng.normal(size=(500, 5)).astype(np.float32)) * 3.0
        labels = torch.as_tensor(logits.argmax(1).numpy())
        flip = rng.random(500) < 0.3
        labels[flip] = torch.as_tensor(rng.integers(0, 5, flip.sum()))
        cal = build_calibrator("temperature", 5).fit(logits, labels)
        assert 0.05 < cal.temperature < 20
        iso = build_calibrator("isotonic", 5)
        p = torch.softmax(logits, 1).numpy()
        iso.fit(p, labels.numpy())
        q = iso.transform(p)
        assert np.allclose(q.sum(1), 1.0, atol=1e-5)
        return f"temperature T={cal.temperature:.3f}; isotonic renormalised OK"
    check("A10 calibrators", t_calib)

    def t_gate():
        a = decision_gate(0.2, 0.9, "accept", 0.85, 0.9)
        b = decision_gate(0.95, 0.9, "accept", 0.85, 0.5)
        c = decision_gate(0.2, 0.1, "retake", 0.85, 0.5)
        assert a[0] == "AI_DECISION" and b[0] == "HUMAN_REVIEW" and c[0] == "RETAKE_IMAGE"
        return f"{a[0]} / {b[0]} / {c[0]}"
    check("A10 decision gate", t_gate)

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'='*64}\nSELF-TEST: {n_ok}/{len(results)} blocks passed\n{'='*64}")
    if n_ok != len(results):
        for n, ok, m in results:
            if not ok:
                print(f"  FAILED: {n} -> {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
