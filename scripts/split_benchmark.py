"""Splits data/evaluation/benchmark.json into a calibration set (~20%) and
a final held-out test set (~80%), stratified by category so both sets have
reasonable representation of every category, using a deterministic random
seed so the split is 100% reproducible.

CRITICAL (per project requirement): the calibration set is for future
threshold selection (Dimension 2+, not implemented yet). The test set MUST
NOT be touched until final evaluation — do not peek at it while tuning
anything.

Usage:
    python scripts/split_benchmark.py \
        --benchmark data/evaluation/benchmark.json \
        --calibration-fraction 0.2 --seed 42
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stratified_split(items, calibration_fraction: float, seed: int):
    by_category = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)

    rng = random.Random(seed)
    calibration, test = [], []
    for category, cat_items in by_category.items():
        cat_items = sorted(cat_items, key=lambda i: i["id"])  # stable order before shuffling
        shuffled = cat_items[:]
        rng.shuffle(shuffled)
        n_calibration = max(1, round(len(shuffled) * calibration_fraction)) if len(shuffled) > 1 else 0
        calibration.extend(shuffled[:n_calibration])
        test.extend(shuffled[n_calibration:])

    return calibration, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/evaluation/benchmark.json")
    parser.add_argument("--out-dir", default="data/evaluation")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.benchmark, "r", encoding="utf-8") as f:
        items = json.load(f)

    calibration, test = stratified_split(items, args.calibration_fraction, args.seed)

    calibration_ids = {i["id"] for i in calibration}
    test_ids = {i["id"] for i in test}
    overlap = calibration_ids & test_ids
    if overlap:
        print(f"ERROR: {len(overlap)} ids appear in both splits: {sorted(overlap)[:10]}")
        sys.exit(1)
    if calibration_ids | test_ids != {i["id"] for i in items}:
        print("ERROR: split does not cover every benchmark item exactly once.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    calibration_path = os.path.join(args.out_dir, "calibration.json")
    test_path = os.path.join(args.out_dir, "test.json")
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    print(f"Calibration set: {len(calibration)} items -> {calibration_path}")
    print(f"Test set:        {len(test)} items -> {test_path}")

    def counts(subset):
        c = defaultdict(int)
        for i in subset:
            c[i["category"]] += 1
        return dict(sorted(c.items()))

    print("\nCategory counts (calibration):", counts(calibration))
    print("Category counts (test):       ", counts(test))
    print("\nReminder: the test set is held out. Do not use it to select any "
          "threshold or hyperparameter — only for final evaluation.")


if __name__ == "__main__":
    main()
