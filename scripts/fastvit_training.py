from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class Sample:
    path: Path
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FastViT data loaders and model for COVIDxCT."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/covidxct_preprocessed"),
        help="Root directory containing preprocessed images.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/covidxct_splits"),
        help="Directory containing train/val/test CSV files.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Split name for training (without .csv).",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val",
        help="Split name for validation (without .csv).",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="test",
        help="Split name for testing (without .csv).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loaders.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loaders.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size expected by the model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="fastvit_t8",
        help="Timm model name for FastViT.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="Number of output classes (0 to infer from labels).",
    )
    return parser.parse_args()


def load_samples(csv_path: Path, data_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(Sample(path=data_root / row["path"], label=row["label"]))
    return samples


def build_label_map(samples: Iterable[Sample]) -> dict[str, int]:
    labels = sorted({sample.label for sample in samples})
    return {label: idx for idx, label in enumerate(labels)}


class CovidXCTDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: list[Sample],
        label_map: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.samples = samples
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        label = self.label_map[sample.label]
        return tensor, label


def create_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader[tuple[torch.Tensor, int]], DataLoader[tuple[torch.Tensor, int]], DataLoader[tuple[torch.Tensor, int]], dict[str, int]]:
    train_csv = args.splits_dir / f"{args.train_split}.csv"
    val_csv = args.splits_dir / f"{args.val_split}.csv"
    test_csv = args.splits_dir / f"{args.test_split}.csv"

    for csv_path in (train_csv, val_csv, test_csv):
        if not csv_path.exists():
            raise SystemExit(f"Missing split file: {csv_path}")

    train_samples = load_samples(train_csv, args.data_root)
    val_samples = load_samples(val_csv, args.data_root)
    test_samples = load_samples(test_csv, args.data_root)

    label_map = build_label_map(train_samples + val_samples + test_samples)
    transform = create_transforms(args.image_size)

    train_dataset = CovidXCTDataset(train_samples, label_map, transform)
    val_dataset = CovidXCTDataset(val_samples, label_map, transform)
    test_dataset = CovidXCTDataset(test_samples, label_map, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, label_map


def create_model(model_name: str, num_classes: int) -> nn.Module:
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise SystemExit(
            f"Data root not found: {args.data_root}. Run preprocessing first."
        )
    if not args.splits_dir.exists():
        raise SystemExit(
            f"Splits directory not found: {args.splits_dir}. Run split generation first."
        )

    train_loader, val_loader, test_loader, label_map = create_dataloaders(args)
    num_classes = args.num_classes or len(label_map)

    model = create_model(args.model_name, num_classes)

    print(
        "Prepared FastViT training components:\n"
        f"- Train batches: {len(train_loader)}\n"
        f"- Val batches: {len(val_loader)}\n"
        f"- Test batches: {len(test_loader)}\n"
        f"- Classes: {num_classes} ({', '.join(label_map)})\n"
        f"- Model: {args.model_name}"
    )
    print(model.__class__.__name__)


if __name__ == "__main__":
    main()
