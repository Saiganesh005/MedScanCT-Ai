from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 3D CT volumes (NIfTI) into 2D slice PNGs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/covidxct_3d"),
        help="Directory containing 3D CT volumes (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/covidxct_slices"),
        help="Directory to write 2D slice PNGs.",
    )
    parser.add_argument(
        "--axis",
        choices=("axial", "coronal", "sagittal"),
        default="axial",
        help="Slice axis to extract (axial=Z, coronal=Y, sagittal=X).",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=-1000.0,
        help="Minimum HU/voxel value for clipping before normalization.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=400.0,
        help="Maximum HU/voxel value for clipping before normalization.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Save every Nth slice (e.g., 2 keeps every other slice).",
    )
    return parser.parse_args()


def normalize_slice(slice_data: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    clipped = np.clip(slice_data, clip_min, clip_max)
    normalized = (clipped - clip_min) / (clip_max - clip_min)
    return (normalized * 255).astype(np.uint8)


def iter_volumes(root: Path) -> list[Path]:
    return sorted(root.glob("*.nii")) + sorted(root.glob("*.nii.gz"))


def axis_to_index(axis: str) -> int:
    return {"sagittal": 0, "coronal": 1, "axial": 2}[axis]


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(
            f"Input directory not found: {args.input_dir}. Provide 3D CT volumes."
        )

    volumes = iter_volumes(args.input_dir)
    if not volumes:
        raise SystemExit(
            f"No NIfTI files found in {args.input_dir}. Expected .nii or .nii.gz files."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    axis_index = axis_to_index(args.axis)

    for volume_path in volumes:
        volume = nib.load(volume_path)
        data = volume.get_fdata()
        if data.ndim != 3:
            raise SystemExit(f"Expected 3D volume, got shape {data.shape} in {volume_path}.")

        num_slices = data.shape[axis_index]
        volume_output = args.output_dir / volume_path.stem
        volume_output.mkdir(parents=True, exist_ok=True)

        for slice_idx in range(0, num_slices, args.stride):
            if axis_index == 0:
                slice_data = data[slice_idx, :, :]
            elif axis_index == 1:
                slice_data = data[:, slice_idx, :]
            else:
                slice_data = data[:, :, slice_idx]

            png_data = normalize_slice(slice_data, args.clip_min, args.clip_max)
            output_path = volume_output / f"slice_{slice_idx:04d}.png"
            imageio.imwrite(output_path, png_data)

        print(f"Wrote {volume_output}")


if __name__ == "__main__":
    main()
