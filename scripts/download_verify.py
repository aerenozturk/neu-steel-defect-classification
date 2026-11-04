"""Step 1: download and audit NEU defect images (CPU only).

Run from project root:
    python scripts/01_download_verify.py
Or after manually extracting a Kaggle zip into data/raw:
    python scripts/01_download_verify.py --skip-download
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUTPUT = ROOT / "outputs"
CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_neu_dir(root: Path) -> Path | None:
    """Locate the class-labelled train/validation directories in nested archives."""
    candidates = [root, *[p for p in root.rglob("NEU-DET") if p.is_dir()]]
    for p in candidates:
        if (p / "train" / "images").is_dir() and (p / "validation" / "images").is_dir():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="Use files manually extracted under data/raw")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    neu = find_neu_dir(DATA)
    if neu is None and not args.skip_download:
        print("Downloading public NEU dataset from Kaggle (about 28 MB)...")
        try:
            import kagglehub
            path = kagglehub.dataset_download(
                "kaustubhdikshit/neu-surface-defect-database",
                output_dir=str(DATA),
            )
            print("Downloaded to:", path)
        except Exception as exc:
            print("Download failed:", exc, file=sys.stderr)
            print("Download the ZIP from Kaggle, extract into data/raw, then rerun with --skip-download.")
            return 1
        neu = find_neu_dir(DATA)

    if neu is None:
        print("Could not find NEU-DET/train/images and NEU-DET/validation/images under", DATA)
        print("Please check the extracted dataset folder structure.")
        return 1

    print("Dataset directory:", neu)
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for split in ("train", "validation"):
        split_dir = neu / split / "images"
        present = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        unexpected = set(present) - set(CLASSES)
        if unexpected:
            errors.append(f"Unknown classes in {split}: {sorted(unexpected)}")
        for cls in CLASSES:
            folder = split_dir / cls
            if not folder.is_dir():
                errors.append(f"Missing class directory: {folder}")
                continue
            paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not paths:
                errors.append(f"No images found: {folder}")
            for image_path in paths:
                try:
                    with Image.open(image_path) as img:
                        img.load()
                        width, height, mode = img.width, img.height, img.mode
                    if (width, height) != (200, 200):
                        errors.append(f"Unexpected image size for {image_path}: {width}x{height}")
                    if mode not in ("L", "P"):
                        errors.append(f"Unexpected image mode for {image_path}: {mode}")
                    digest = sha256(image_path.read_bytes()).hexdigest()
                    relpath = image_path.relative_to(ROOT).as_posix()
                    rows.append({
                        "path": relpath,
                        "split": split,
                        "label": cls,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "sha256": digest,
                    })
                    hashes[digest].append((split, cls, relpath))
                except (OSError, UnidentifiedImageError) as exc:
                    errors.append(f"Unreadable image {image_path}: {exc}")

    df = pd.DataFrame(rows, columns=["path", "split", "label", "width", "height", "mode", "sha256"])
    if df.empty:
        print("No readable images were found.")
        return 1
    counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=CLASSES, columns=["train", "validation"], fill_value=0)
    counts["total"] = counts.sum(axis=1)
    print("\nCLASS COUNTS\n")
    print(counts.to_string())
    print("\nTOTAL:", len(df), "| train:", int((df.split == "train").sum()), "| validation:", int((df.split == "validation").sum()))

    repeated = [entries for entries in hashes.values() if len(entries) > 1]
    cross_split = [entries for entries in repeated if len({e[0] for e in entries}) > 1]
    cross_label = [entries for entries in repeated if len({e[1] for e in entries}) > 1]
    print("Exact duplicate image groups:", len(repeated))
    print("Exact duplicates crossing train/validation:", len(cross_split))
    print("Exact duplicates with conflicting labels:", len(cross_label))
    if cross_split:
        print("WARNING: Overlapping images across held-out test and train! Resolve before modeling.")
        for group in cross_split[:5]:
            print("  ", group)
    if cross_label:
        print("WARNING: Some exact duplicate images have conflicting labels.")

    expected = (len(df) == 1800 and (counts["train"] == 240).all() and (counts["validation"] == 60).all())
    print("Expected 1,800 images and 240/60 per class:", "PASS" if expected else "CHECK")
    if errors:
        print("\nIMAGE / FOLDER WARNINGS (first 15):")
        for msg in errors[:15]:
            print(" -", msg)

    df.to_csv(OUTPUT / "step1_dataset_manifest.csv", index=False)
    summary = {
        "source": "https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database",
        "dataset_directory": str(neu),
        "task": "Six-class steel surface defect image classification",
        "split_policy": "train: 5-fold CV; validation directory: held-out final test only",
        "total_images": int(len(df)),
        "counts": {str(k): {str(s): int(v) for s, v in row.items()} for k, row in counts.to_dict(orient="index").items()},
        "exact_duplicate_groups": len(repeated),
        "exact_cross_split_duplicate_groups": len(cross_split),
        "exact_cross_label_duplicate_groups": len(cross_label),
        "expected_counts_pass": bool(expected),
        "warnings": errors[:100],
    }
    (OUTPUT / "step1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved:", OUTPUT / "step1_dataset_manifest.csv")
    print("Saved:", OUTPUT / "step1_summary.json")
    if not expected or errors or cross_split or cross_label:
        print("DATASET AUDIT NEEDS REVIEW before feature extraction.")
        return 2
    print("DATASET AUDIT PASSED. Step 1 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
