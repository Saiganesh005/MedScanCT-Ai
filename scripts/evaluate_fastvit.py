from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import timm
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class Sample:
    path: Path
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FastViT on COVIDxCT and generate visualizations."
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
        "--test-split",
        type=str,
        default="test",
        help="Split name for testing (without .csv).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the test data loader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for the data loader.",
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("fastvit_covid_ct.pth"),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fastvit_eval"),
        help="Directory to write evaluation artifacts.",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate a Grad-CAM visualization for the first test sample.",
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
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def load_model(model_name: str, num_classes: int, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def save_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - FastViT")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    labels_bin = label_binarize(labels, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 6))

    for idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels_bin[:, idx], probs[:, idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - FastViT")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_classification_report(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    report = classification_report(labels, preds, target_names=class_names)
    output_path.write_text(report, encoding="utf-8")


def save_results_csv(
    labels: np.ndarray,
    preds: np.ndarray,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", "predicted_label"])
        writer.writerows(zip(labels.tolist(), preds.tolist()))


def save_gradcam(
    model: torch.nn.Module,
    sample: torch.Tensor,
    class_names: list[str],
    output_path: Path,
) -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: grad-cam. Install with `pip install grad-cam`."
        ) from exc

    target_layer = getattr(model, "blocks", None)
    if not target_layer:
        raise SystemExit("Unable to locate model.blocks for Grad-CAM.")

    cam = GradCAM(model=model, target_layers=[target_layer[-1]])
    grayscale_cam = cam(input_tensor=sample)[0]

    sample_np = sample.cpu().numpy()[0].transpose(1, 2, 0)
    sample_np = (sample_np - sample_np.min()) / (sample_np.max() - sample_np.min())
    visualization = show_cam_on_image(sample_np, grayscale_cam, use_rgb=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(visualization)
    ax.axis("off")
    ax.set_title(f"Grad-CAM - {class_names[0] if class_names else 'sample'}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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

    test_csv = args.splits_dir / f"{args.test_split}.csv"
    if not test_csv.exists():
        raise SystemExit(f"Missing split file: {test_csv}")

    samples = load_samples(test_csv, args.data_root)
    label_map = build_label_map(samples)
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]

    transform = create_transforms(args.image_size)
    dataset = CovidXCTDataset(samples, label_map, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = args.num_classes or len(label_map)
    model = load_model(args.model_name, num_classes, args.checkpoint, device)

    labels, preds, probs = evaluate_model(model, loader, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(labels, preds, class_names, args.output_dir / "confusion_matrix.png")
    save_roc_curve(labels, probs, class_names, args.output_dir / "roc_curve.png")
    save_classification_report(
        labels, preds, class_names, args.output_dir / "classification_report.txt"
    )
    save_results_csv(labels, preds, args.output_dir / "fastvit_results.csv")

    if args.gradcam:
        sample, _ = next(iter(loader))
        save_gradcam(
            model,
            sample[:1].to(device),
            class_names,
            args.output_dir / "gradcam.png",
        )

    print(f"Evaluation artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
