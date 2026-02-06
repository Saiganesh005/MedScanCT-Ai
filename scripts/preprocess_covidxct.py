from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess COVIDxCT 2D images (resize + grayscale/RGB normalization)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/covidxct"),
        help="Directory containing input 2D images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/covidxct_preprocessed"),
        help="Directory to write preprocessed images.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Target square size for resizing (e.g., 224).",
    )
    parser.add_argument(
        "--mode",
        choices=("L", "RGB"),
        default="L",
        help="Color mode for output images (L=grayscale, RGB=color).",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        help="File extensions to include as samples.",
    )
    return parser.parse_args()


def iter_images(root: Path, extensions: list[str]) -> list[Path]:
    normalized_extensions = {ext.lower() for ext in extensions}
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]


def preprocess_image(path: Path, size: int, mode: str) -> Image.Image:
    with Image.open(path) as image:
        image_converted = image.convert(mode)
        return image_converted.resize((size, size), resample=Image.BILINEAR)


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(
            f"Input directory not found: {args.input_dir}. Provide 2D images first."
        )

    images = iter_images(args.input_dir, args.extensions)
    if not images:
        raise SystemExit(
            f"No images found under {args.input_dir}. Check the dataset path."
        )

    for image_path in images:
        relative_path = image_path.relative_to(args.input_dir)
        output_path = args.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        processed = preprocess_image(image_path, args.size, args.mode)
        processed.save(output_path)

    print(
        f"Preprocessed {len(images)} images into {args.output_dir} "
        f"(size={args.size}, mode={args.mode})."
    )


if __name__ == "__main__":
    main()
