#!/usr/bin/env python3
"""Step 06 - Deployment Optimization block.

    knowledge distillation -> student model -> FP16 / INT8 quantization
    -> ONNX export -> latency + size benchmark

The student is image-only (no anatomy maps, no MoE) so it can run on an edge
device; it learns from the teacher's soft ordinal distribution plus the real
labels.  Everything reported here is measured on this machine, not quoted.

    python scripts/06_deploy.py --epochs 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import CACHE_DIR, OUT_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.data.eyepacs import CachedEyePACS  # noqa: E402
from dr.metrics import diagnostic_metrics, ordinal_metrics  # noqa: E402
from dr.model import build_model  # noqa: E402
from dr.sampling import HardExampleState, make_sampler  # noqa: E402
from dr.modules.a6_ordinal import class_balanced_focal_loss  # noqa: E402
from dr.modules.a4_backbone_lora import inject_lora  # noqa: E402


@torch.no_grad()
def feature_dim(backbone: nn.Module, img_size: int = 224) -> int:
    """Actual pooled feature width, measured with a dummy forward."""
    was = backbone.training
    backbone.eval()
    d = int(backbone(torch.zeros(1, 3, img_size, img_size)).shape[-1])
    backbone.train(was)
    return d


class StudentNet(nn.Module):
    """Compact image-only grader distilled from the full pipeline."""

    def __init__(self, name: str = "mobilenetv3_small_100", n_grades: int = 5,
                 img_size: int = 224):
        super().__init__()
        import timm
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        # `num_features` is not always the pooled output width (mobilenetv3
        # reports 576 but emits 1024 through its conv-head), so probe it.
        d = feature_dim(self.backbone, img_size)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1),
                                  nn.Linear(d, n_grades))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def model_size_mb(m: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6


@torch.no_grad()
def bench(fn, x, n=30, warmup=5) -> float:
    for _ in range(warmup):
        fn(x)
    t = time.time()
    for _ in range(n):
        fn(x)
    return (time.time() - t) / n * 1000.0


@torch.no_grad()
def student_metrics(student, loader, device) -> dict:
    student.eval()
    P, Y = [], []
    for b in loader:
        x = b["image"].to(device)
        P.append(torch.softmax(student(x), 1).float().cpu().numpy())
        Y.append(b["grade"].numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return {**diagnostic_metrics(y, p.argmax(1), p), **ordinal_metrics(y, p.argmax(1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=OUT_DIR / "retfound_plus_laft_xai" / "best.pt")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--student", default="mobilenetv3_small_100")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.6, help="weight on the KD term")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = default_config(); cfg.device = args.device
    device = cfg.torch_device()
    out_dir = args.ckpt.parent

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("config", {})
    if isinstance(saved, dict) and "backbone" in saved:
        cfg.backbone, cfg.fusion, cfg.moe = saved["backbone"], saved["fusion"], saved["moe"]
    teacher = build_model(cfg)
    inject_lora(teacher.backbone.vit, (ck.get("gla_lora") or ck.get("allora"))["ranks"], cfg.backbone)
    teacher.load_state_dict(ck["model"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    train_ds = CachedEyePACS(args.cache, "train", augment=True, cfg=cfg)
    test_ds = CachedEyePACS(args.cache, "test", augment=False, cfg=cfg)
    if args.max_train and args.max_train < len(train_ds):
        train_ds.idx = np.random.default_rng(0).choice(train_ds.idx, args.max_train,
                                                       replace=False)
    # The teacher was trained under an n^-0.5 sampler and a class-balanced focal
    # loss.  Distilling it under a uniform shuffle and a plain cross-entropy puts
    # the student back in front of a 74%-"No DR" distribution with no correction,
    # and it collapses to predicting the majority class: the previous student
    # scored 0.738041 against a majority-class rate of 0.738041, i.e. exactly
    # constant.  Reuse the teacher's own re-balancing here.
    counts = train_ds.class_counts()
    hard = HardExampleState.empty(len(train_ds.meta))
    sampler = make_sampler(train_ds.labels(), train_ds.meta_indices(), hard,
                           cfg.sampling, len(train_ds))
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, drop_last=True)
    counts_t = torch.as_tensor(counts)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=0)

    print(f"[deploy] distilling {args.student} from the full pipeline "
          f"on {len(train_ds)} images")
    student = StudentNet(args.student).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01)
    T = args.temperature

    for ep in range(args.epochs):
        student.train()
        tot, n = 0.0, 0
        for step, b in enumerate(train_ld):
            x = b["image"].to(device)
            a = b["anatomy"].to(device)
            y = b["grade"].to(device)
            with torch.no_grad():
                t_out = teacher({k: v.to(device) for k, v in b.items()
                                 if k in ("image", "crops", "anatomy", "quality_axes", "hardness")})
                t_logits = t_out["ordinal"].logits_ce
            s_logits = student(x)
            kd = F.kl_div(F.log_softmax(s_logits / T, 1),
                          F.softmax(t_logits / T, 1),
                          reduction="batchmean") * (T * T)
            ce = class_balanced_focal_loss(
                s_logits, y, counts_t.to(device), cfg.loss.cb_beta,
                cfg.loss.focal_gamma, weight_power=cfg.sampling.class_weight_power)
            loss = args.alpha * kd + (1 - args.alpha) * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(y); n += len(y)
            if step % 40 == 0:
                print(f"  ep{ep+1} step {step}/{len(train_ld)} loss={tot/n:.4f}",
                      flush=True)
        print(f"[deploy] epoch {ep+1}: distill loss = {tot/max(n,1):.4f}")

    # ---- accuracy of teacher vs student ---------------------------------
    print("[deploy] measuring teacher and student on the test split")
    t_P, t_Y = [], []
    with torch.no_grad():
        for b in test_ld:
            o = teacher({k: v.to(device) for k, v in b.items()
                     if k in ("image", "crops", "anatomy", "quality_axes", "hardness")})
            t_P.append(o["ordinal"].class_probs.float().cpu().numpy())
            t_Y.append(b["grade"].numpy())
    tp, ty = np.concatenate(t_P), np.concatenate(t_Y)
    tm = {**diagnostic_metrics(ty, tp.argmax(1), tp), **ordinal_metrics(ty, tp.argmax(1))}
    sm = student_metrics(student, test_ld, device)
    retention = sm["accuracy"] / tm["accuracy"] if tm["accuracy"] > 0 else float("nan")

    # ---- quantization + ONNX ---------------------------------------------
    S = cfg.preproc.global_size
    x_cpu = torch.randn(1, 3, S, S)
    student_cpu = StudentNet(args.student)
    student_cpu.load_state_dict(student.state_dict())
    student_cpu.eval().cpu()

    variants = {}
    variants["student_fp32"] = {
        "size_mb": model_size_mb(student_cpu),
        "latency_ms": bench(lambda z: student_cpu(z), x_cpu),
    }
    fp16 = StudentNet(args.student); fp16.load_state_dict(student.state_dict())
    fp16 = fp16.eval().half()
    variants["student_fp16"] = {
        "size_mb": model_size_mb(fp16),
        "latency_ms": bench(lambda z: fp16(z.half()), x_cpu),
    }
    try:
        # ARM/Apple Silicon needs the qnnpack engine; fbgemm is x86-only and
        # raises "Didn't find engine for operation" here.
        for eng in ("qnnpack", "fbgemm"):
            if eng in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = eng
                break
        int8 = torch.ao.quantization.quantize_dynamic(
            student_cpu, {nn.Linear}, dtype=torch.qint8)
        variants["student_int8_dynamic"] = {
            "size_mb": model_size_mb(int8),
            "latency_ms": bench(lambda z: int8(z), x_cpu),
        }
    except Exception as e:  # noqa: BLE001
        variants["student_int8_dynamic"] = {"error": str(e)}

    onnx_path = out_dir / "student.onnx"
    try:
        torch.onnx.export(student_cpu, x_cpu, str(onnx_path),
                          input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17)
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        arr = x_cpu.numpy()
        lat = bench(lambda z: sess.run(None, {"image": arr}), arr)
        # torch.onnx writes weights to a sibling .onnx.data file; the graph
        # alone is ~0.3 MB and would badly understate the deployed footprint.
        onnx_bytes = onnx_path.stat().st_size + sum(
            p.stat().st_size for p in onnx_path.parent.glob(onnx_path.name + ".data"))
        variants["student_onnx_cpu"] = {
            "size_mb": onnx_bytes / 1e6, "latency_ms": lat}
    except Exception as e:  # noqa: BLE001
        variants["student_onnx_cpu"] = {"error": str(e)}

    teacher_size = model_size_mb(teacher)

    # Guard against a silently collapsed student.  On a 74%-"No DR" test split a
    # constant predictor scores 0.738 accuracy and ~0.965 "accuracy retention",
    # which reads as a success; only balanced accuracy and QWK expose it.  Report
    # the majority-class rate alongside, and say plainly when the student has not
    # beaten it.
    test_labels = np.asarray([int(test_ds.meta.iloc[int(j)]["grade"])
                              for j in test_ds.idx])
    majority_rate = float(np.bincount(test_labels, minlength=5).max() / len(test_labels))
    collapsed = bool(sm["accuracy"] <= majority_rate + 1e-4)
    if collapsed:
        print(f"[deploy] WARNING: student accuracy {sm['accuracy']:.6f} does not beat "
              f"the majority-class rate {majority_rate:.6f} - it has collapsed to a "
              f"constant predictor. Balanced accuracy {sm['balanced_accuracy']:.4f}.")

    result = {
        "teacher": {"size_mb": teacher_size, **tm},
        "student": {"name": args.student, **sm},
        "majority_class_rate": majority_rate,
        "student_beats_majority": not collapsed,
        "student_accuracy_over_majority": float(sm["accuracy"] - majority_rate),
        "accuracy_retention": float(retention),
        "qwk_retention": float(sm["quadratic_weighted_kappa"]
                               / tm["quadratic_weighted_kappa"])
        if tm["quadratic_weighted_kappa"] > 0 else float("nan"),
        "compression_ratio": float(teacher_size / variants["student_fp32"]["size_mb"]),
        "variants": variants,
    }
    metrics_path = out_dir / "distilled_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2))
    save_csv_alongside(result, metrics_path)
    save_csv_alongside(variants, metrics_path,
                       csv_path=metrics_path.with_name("distilled_variants.csv"))
    torch.save(student.state_dict(), out_dir / "student.pt")

    print("\n" + "=" * 68)
    print("DEPLOYMENT OPTIMIZATION RESULTS")
    print("=" * 68)
    print(f"Teacher  size={teacher_size:8.2f} MB  acc={tm['accuracy']:.4f}  "
          f"QWK={tm['quadratic_weighted_kappa']:.4f}")
    print(f"Student  ({args.student})          acc={sm['accuracy']:.4f}  "
          f"QWK={sm['quadratic_weighted_kappa']:.4f}")
    print(f"Accuracy retention: {retention*100:.2f}%   "
          f"Compression: {result['compression_ratio']:.1f}x")
    print(f"\n{'variant':24s} {'size MB':>10s} {'latency ms':>12s}")
    for k, v in variants.items():
        if "error" in v:
            print(f"{k:24s} {'-':>10s} {'FAILED: ' + v['error'][:30]:>12s}")
        else:
            print(f"{k:24s} {v['size_mb']:10.2f} {v['latency_ms']:12.2f}")
    print("=" * 68)
    print(f"[saved] {out_dir/'distilled_metrics.json'}")


if __name__ == "__main__":
    main()
