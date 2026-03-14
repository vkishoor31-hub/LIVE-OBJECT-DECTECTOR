"""
download_dataset.py — Galaxy Scan Animal Dataset Downloader
============================================================
Downloads animal images from the COCO dataset and converts
annotations to YOLO format for fine-tuning.

Usage:
    python download_dataset.py

Output:
    dataset/
        images/train/   ← training images
        images/val/     ← validation images
        labels/train/   ← YOLO .txt labels (train)
        labels/val/     ← YOLO .txt labels (val)
        animals.yaml    ← YOLO dataset config
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Try to import optional helpers ────────────────────────────────────────────
try:
    from tqdm import tqdm  # type: ignore[import]
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent / "dataset"
IMG_TRAIN = BASE_DIR / "images" / "train"
IMG_VAL   = BASE_DIR / "images" / "val"
LBL_TRAIN = BASE_DIR / "labels" / "train"
LBL_VAL   = BASE_DIR / "labels" / "val"

COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_URL   = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL   = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# COCO category names we want → mapped to YOLO class index
ANIMAL_CATEGORIES: Dict[str, int] = {
    "bird":     0,
    "cat":      1,
    "dog":      2,
    "horse":    3,
    "sheep":    4,
    "cow":      5,
    "elephant": 6,
    "bear":     7,
    "zebra":    8,
    "giraffe":  9,
}

# Max images per category (keeps dataset manageable on a laptop)
MAX_PER_CLASS_TRAIN = 500
MAX_PER_CLASS_VAL   = 100


# ── Helpers ───────────────────────────────────────────────────────────────────

def _makedirs() -> None:
    for d in (IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL):
        d.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dest: Path) -> None:
    """Download url → dest with a simple progress indicator."""
    print(f"  Downloading {url.split('/')[-1]} …")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved → {dest}")


def _extract(archive: Path, out_dir: Path) -> None:
    import zipfile
    print(f"  Extracting {archive.name} …")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(out_dir)
    archive.unlink()
    print("  Done.")


def _download_and_extract(url: str, out_dir: Path) -> None:
    archive = out_dir / url.split("/")[-1]
    _download_file(url, archive)
    _extract(archive, out_dir)


def _load_coco_json(path: Path) -> Tuple[Dict[int, int], Dict[int, List[Any]], Dict[int, Any]]:
    """
    Returns:
        cat_id_to_yolo   : {coco_cat_id → yolo_class_idx}
        img_to_anns      : {image_id → [annotation, …]}
        id_to_img_info   : {image_id → image_info_dict}
    """
    print(f"  Loading {path.name} …")
    with open(path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)

    # Build category map
    cat_id_to_yolo: Dict[int, int] = {}
    for cat in data.get("categories", []):
        name: str = cat["name"]
        if name in ANIMAL_CATEGORIES:
            cat_id_to_yolo[cat["id"]] = ANIMAL_CATEGORIES[name]

    # Build image-id → annotations index
    img_to_anns: Dict[int, List[Any]] = {}
    for ann in data.get("annotations", []):
        if ann["category_id"] not in cat_id_to_yolo:
            continue
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Build image-id → image info
    id_to_img_info: Dict[int, Any] = {
        img["id"]: img for img in data.get("images", [])
    }

    return cat_id_to_yolo, img_to_anns, id_to_img_info


def _coco_bbox_to_yolo(
    bbox: List[float], img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Convert COCO [x, y, w, h] → YOLO [cx, cy, w, h] (normalised)."""
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return None
    cx = (x + bw / 2.0) / img_w
    cy = (y + bh / 2.0) / img_h
    nw = bw / img_w
    nh = bh / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def _build_split(
    split_name: str,
    ann_path: Path,
    src_imgs_dir: Path,
    dst_imgs_dir: Path,
    dst_lbls_dir: Path,
    max_per_class: int,
) -> int:
    """Convert one COCO split (train/val) to YOLO format."""
    cat_id_to_yolo, img_to_anns, id_to_img_info = _load_coco_json(ann_path)

    # Limit images per class
    class_counts: Dict[int, int] = {v: 0 for v in ANIMAL_CATEGORIES.values()}
    selected_ids: Set[int] = set()

    for img_id, anns in img_to_anns.items():
        for ann in anns:
            yolo_cls: int = cat_id_to_yolo[ann["category_id"]]
            if class_counts[yolo_cls] < max_per_class:
                selected_ids.add(img_id)
                break

    total = 0
    it: Any = selected_ids
    if HAS_TQDM:
        it = tqdm(selected_ids, desc=f"  [{split_name}]", ncols=80)

    for img_id in it:
        info: Dict[str, Any] = id_to_img_info.get(img_id, {})
        if not info:
            continue

        file_name: str = info["file_name"]
        img_w: int = int(info["width"])
        img_h: int = int(info["height"])
        src_img: Path = src_imgs_dir / file_name

        if not src_img.exists():
            continue  # image may not have been downloaded yet; skip

        # Write label file
        lines: List[str] = []
        for ann in img_to_anns.get(img_id, []):
            yolo_cls = cat_id_to_yolo.get(ann["category_id"], -1)
            if yolo_cls < 0:
                continue
            result = _coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            if result:
                cx, cy, nw, nh = result
                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                class_counts[yolo_cls] += 1

        if not lines:
            continue

        stem: str = Path(file_name).stem
        dst_img: Path = dst_imgs_dir / file_name
        dst_lbl: Path = dst_lbls_dir / f"{stem}.txt"

        shutil.copy(src_img, dst_img)
        dst_lbl.write_text("\n".join(lines), encoding="utf-8")
        total += 1

    print(f"  [{split_name}] {total} images written.")
    return total


def _write_yaml() -> None:
    yaml_path = BASE_DIR / "animals.yaml"
    names = list(ANIMAL_CATEGORIES.keys())
    lines = [
        f"path: {BASE_DIR.as_posix()}",
        f"train: images/train",
        f"val:   images/val",
        "",
        f"nc: {len(names)}",
        f"names: {names}",
        "",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Dataset config → {yaml_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Galaxy Scan — COCO Animal Dataset Downloader")
    print("=" * 60)

    _makedirs()

    tmp = BASE_DIR / "_tmp"
    tmp.mkdir(exist_ok=True)

    # ── Download & extract COCO images ────────────────────────────────────
    train_imgs_dir = tmp / "train2017"
    val_imgs_dir   = tmp / "val2017"
    ann_dir        = tmp / "annotations"

    if not train_imgs_dir.exists():
        print("\n[1/3] Downloading COCO train images (this may take a while)…")
        _download_and_extract(COCO_TRAIN_URL, tmp)
    else:
        print("\n[1/3] COCO train images already present — skipping download.")

    if not val_imgs_dir.exists():
        print("\n[2/3] Downloading COCO val images…")
        _download_and_extract(COCO_VAL_URL, tmp)
    else:
        print("\n[2/3] COCO val images already present — skipping download.")

    if not ann_dir.exists():
        print("\n[3/3] Downloading COCO annotations…")
        _download_and_extract(COCO_ANN_URL, tmp)
    else:
        print("\n[3/3] COCO annotations already present — skipping download.")

    # ── Convert to YOLO format ────────────────────────────────────────────
    print("\n[4/4] Converting annotations to YOLO format…")
    _build_split(
        "train",
        ann_dir / "instances_train2017.json",
        train_imgs_dir,
        IMG_TRAIN,
        LBL_TRAIN,
        MAX_PER_CLASS_TRAIN,
    )
    _build_split(
        "val",
        ann_dir / "instances_val2017.json",
        val_imgs_dir,
        IMG_VAL,
        LBL_VAL,
        MAX_PER_CLASS_VAL,
    )

    _write_yaml()

    print("\n✅  Dataset ready!  Run  python train_model.py  to fine-tune.")
    print(f"    Location: {BASE_DIR}")


if __name__ == "__main__":
    main()
