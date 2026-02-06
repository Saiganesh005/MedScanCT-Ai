from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check class balance for COVIDxCT CSV splits."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/covidxct_splits"),
        help="Directory containing train/val/test CSV files.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        help="Split names to inspect (without .csv).",
    )
    return parser.parse_args()


def load_labels(csv_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts[row["label"]] += 1
    return counts


def main() -> None:
    args = parse_args()
    if not args.splits_dir.exists():
        raise SystemExit(
            f"Splits directory not found: {args.splits_dir}. Run the split script first."
        )

    for split_name in args.splits:
        csv_path = args.splits_dir / f"{split_name}.csv"
        if not csv_path.exists():
            raise SystemExit(f"Missing split file: {csv_path}")

        counts = load_labels(csv_path)
        total = sum(counts.values())

        print(f"Split: {split_name} ({total} samples)")
        for label, count in counts.most_common():
            ratio = count / total if total else 0
            print(f"  {label}: {count} ({ratio:.2%})")


if __name__ == "__main__":
    main()
