from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/validation/test CSV splits for the COVIDxCT dataset."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/covidxct"),
        help="Root directory of the downloaded COVIDxCT dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/covidxct_splits"),
        help="Directory to write split CSV files.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of samples assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of samples assigned to the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of samples assigned to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        help="File extensions to include as samples.",
    )
    return parser.parse_args()


def iter_samples(root: Path, extensions: Sequence[str]) -> Iterable[Path]:
    normalized_extensions = {ext.lower() for ext in extensions}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in normalized_extensions:
            yield path


def write_split_csv(output_path: Path, rows: Iterable[tuple[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise SystemExit(
            f"Dataset root not found: {args.data_root}. Run the download script first."
        )

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit("Train/val/test ratios must sum to 1.0.")

    samples = list(iter_samples(args.data_root, args.extensions))
    if not samples:
        raise SystemExit(
            f"No samples found under {args.data_root}. Check the dataset path."
        )

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    total = len(samples)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    for split_name, split_samples in splits.items():
        rows = [
            (str(path.relative_to(args.data_root)), path.parent.name)
            for path in split_samples
        ]
        write_split_csv(args.output_dir / f"{split_name}.csv", rows)

    print(
        "Generated splits with counts: "
        + ", ".join(f"{name}={len(paths)}" for name, paths in splits.items())
    )
    print(f"Split files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
