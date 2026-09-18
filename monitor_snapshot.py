#!/usr/bin/env python3
"""monitor_snapshot.py - one monitoring tick: append current training state
to data/training_monitor.csv and print a compact summary + a stage-1-done
flag. Meant to be re-run every 30-60 minutes during the long training run
(no daemon - deliberately a single snapshot each call, invoked periodically).

    .venv\\Scripts\\python.exe monitor_snapshot.py
"""
from __future__ import annotations

import csv
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "data" / "_resume_pipeline_full.log"
CSV_PATH = ROOT / "data" / "training_monitor.csv"

STEP_RE = re.compile(
    r"^  (s(\d)e(\d+)) step (\d+)/(\d+) loss=([\d.]+).*?([\d.]+) img/s", re.M)
EPOCH_RE = re.compile(
    r"^\[(\w+) e(\d+)\] loss=([\d.]+) \| QWK ([\-\d.]+) F1 ([\d.]+) minRec ([\d.]+) "
    r"\| score ([\-\d.]+) \| acc ([\d.]+)", re.M)
CKPT_RE = re.compile(r"\[checkpoint\] stage (\d+) epoch (\d+) saved")


def _gpu_stats() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,"
                          "memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        util, temp, used, total = [x.strip() for x in out.split(",")]
        return {"util": util, "temp": temp, "mem_used": used, "mem_total": total}
    except Exception as e:                                    # noqa: BLE001
        return {"util": "", "temp": "", "mem_used": "", "mem_total": "", "error": str(e)}


def main() -> None:
    text = LOG.read_text(errors="ignore") if LOG.exists() else ""
    steps = list(STEP_RE.finditer(text))
    epochs = list(EPOCH_RE.finditer(text))
    ckpts = list(CKPT_RE.finditer(text))

    last_step = steps[-1] if steps else None
    last_epoch = epochs[-1] if epochs else None
    gpu = _gpu_stats()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": last_step.group(2) if last_step else "",
        "epoch_in_stage": last_step.group(3) if last_step else "",
        "step": last_step.group(4) if last_step else "",
        "total_steps": last_step.group(5) if last_step else "",
        "train_loss": last_step.group(6) if last_step else "",
        "val_loss": "N/A - not computed by this codebase (see val_qwk)",
        "val_qwk": last_epoch.group(4) if last_epoch else "",
        "gpu_util_pct": gpu["util"], "gpu_temp_c": gpu["temp"],
        "gpu_mem_used_mib": gpu["mem_used"], "gpu_mem_total_mib": gpu["mem_total"],
        "throughput_img_s": last_step.group(7) if last_step else "",
        "note": "",
    }

    is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)

    print(f"[monitor] {row['timestamp']}  stage={row['stage']} "
          f"epoch={row['epoch_in_stage']} step={row['step']}/{row['total_steps']} "
          f"train_loss={row['train_loss']} val_qwk={row['val_qwk']} "
          f"gpu={row['gpu_util_pct']}% {row['gpu_temp_c']}C "
          f"{row['gpu_mem_used_mib']}/{row['gpu_mem_total_mib']}MiB "
          f"throughput={row['throughput_img_s']} img/s")

    stage1_done = any(int(m.group(1)) == 1 and int(m.group(2)) == 5 for m in ckpts)
    print(f"[monitor] stage 1 (5 epochs) complete: {stage1_done}")


if __name__ == "__main__":
    main()
