from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET_NAME = "hgunraj/covidxct"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    destination_root = repo_root / "data" / "covidxct"
    destination_root.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(kagglehub.dataset_download(DATASET_NAME))
    print(f"Dataset downloaded to cache: {dataset_path}")

    if destination_root.resolve() == dataset_path.resolve():
        print("Dataset already located in repository data directory.")
        return

    if any(destination_root.iterdir()):
        print(
            "Destination directory is not empty. "
            "Clear it first if you want a fresh copy."
        )
        return

    shutil.copytree(dataset_path, destination_root, dirs_exist_ok=True)
    print(f"Dataset copied into repository: {destination_root}")


if __name__ == "__main__":
    main()
