#!/usr/bin/env python3
"""Build the svcb-oracle sanity lane: copy every ground_truth.mp4 from the
official data into the evaluator's expected outputs layout.

The oracle lane feeds the benchmark's own reference videos through the
unmodified evaluator. Its score is the effective ceiling of the kit
(0.9977 on 2026-09-18; 370/500 instances exactly 1.0, minimum 0.918).
Score it like any lane:

    python3 bench/make_oracle_lane.py
    <venv>/bin/python VBVR-Pro-Bench/run_evaluation_video.py --device cpu \
        --model_path bench/outputs/svcb-oracle \
        --gt_base ../vbvr-pro-bench-data/VBVR-Pro-Bench-Video \
        --output_dir bench/results/svcb-oracle

The 2026-09-13 release run of this lane scored 0.004 because the videos
were never laid out this way: 498/500 instances came back "no prediction".
"""

import argparse
import shutil
from pathlib import Path

SPLITS = ["In-Domain_50", "Out-of-Domain_50"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="../vbvr-pro-bench-data/VBVR-Pro-Bench-Video",
                    help="official data root (README step 1 location)")
    ap.add_argument("--out", default="bench/outputs/svcb-oracle",
                    help="oracle lane output directory")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    n = 0
    for split in SPLITS:
        for task in sorted((data / split).iterdir()):
            if not task.is_dir():
                continue
            for inst in sorted(task.iterdir()):
                gt = inst / "ground_truth.mp4"
                if gt.is_file():
                    dest = out / split / task.name
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(gt, dest / f"{inst.name}.mp4")
                    n += 1
    print(f"copied {n} ground-truth videos into {out}")


if __name__ == "__main__":
    main()
