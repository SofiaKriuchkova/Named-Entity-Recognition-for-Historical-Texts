#!/usr/bin/env python3
"""tune_hparams.py

Simple grid search over learning rate and batch size using dev F1.
Writes a CSV/JSON summary to --output_dir and prints best config.

Uses train_ner.py logic (self-contained here).
"""

import os
import json
import csv
import argparse
import itertools
import subprocess
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--learning_rates", default="1e-5,5e-6,1e-6,5e-5")
    ap.add_argument("--batch_sizes", default="4,8,16")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max_steps", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lrs = [float(x) for x in args.learning_rates.split(",") if x.strip()]
    bss = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    results = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for lr, bs in itertools.product(lrs, bss):
        run_dir = os.path.join(args.output_dir, f"lr{lr}_bs{bs}")
        cmd = [
            "python", os.path.join(os.path.dirname(__file__), "train_ner.py"),
            "--model_name", args.model_name,
            "--train", args.train,
            "--dev", args.dev,
            "--output_dir", run_dir,
            "--learning_rate", str(lr),
            "--batch_size", str(bs),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--max_steps", str(args.max_steps),
        ]
        if args.fp16:
            cmd.append("--fp16")

        print("\n=== RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        dev = m.get("dev", {})
        f1 = dev.get("eval_f1", dev.get("f1", 0.0))
        results.append({"learning_rate": lr, "batch_size": bs, "f1": float(f1), "run_dir": run_dir})

    # save
    json_path = os.path.join(args.output_dir, f"tuning_results_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.output_dir, f"tuning_results_{stamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["learning_rate", "batch_size", "f1", "run_dir"])
        w.writeheader()
        for r in sorted(results, key=lambda x: x["f1"], reverse=True):
            w.writerow(r)

    best = max(results, key=lambda x: x["f1"])
    print("\nBEST:", best)
    print("Saved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
