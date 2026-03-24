"""
convert_annotations.py
-----------------------
Converts COCO JSON annotations to YOLO .txt label format.

Place this file at the PROJECT ROOT (same level as data.yaml, main.py).

Run ONCE before training:
    python convert_annotations.py

After running, your dataset/ folder will look like:
    dataset/
        train/
            images/
            labels/       <- created by this script
        valid/
            images/
            labels/       <- created by this script
        test/
            images/
            labels/       <- created by this script
"""

import json
from pathlib import Path


# ── Config: paths relative to project root ────────────────────────────────────

DATASET_ROOT = Path("dataset")
SPLITS = ["train", "valid", "test"]


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_split(split: str) -> None:
    split_dir  = DATASET_ROOT / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    # Roboflow exports put the JSON inside images/ — check both locations
    coco_path = images_dir / "_annotations.coco.json"
    if not coco_path.exists():
        coco_path = split_dir / "_annotations.coco.json"

    if not coco_path.exists():
        print(f"[SKIP] No _annotations.coco.json found in {split_dir} or {images_dir}")
        return

    print(f"[{split}] Reading annotations from: {coco_path}")
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_path, "r") as f:
        coco = json.load(f)

    # image_id -> image info
    image_lookup = {img["id"]: img for img in coco["images"]}

    # image_id -> list of annotations
    ann_by_image: dict = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    empty     = 0

    for image_id, image_info in image_lookup.items():
        img_w     = image_info["width"]
        img_h     = image_info["height"]
        file_stem = Path(image_info["file_name"]).stem

        annotations = ann_by_image.get(image_id, [])
        lines = []

        for ann in annotations:
            x_min, y_min, box_w, box_h = ann["bbox"]

            # Clamp to image bounds (guards against occasional bad annotations)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            box_w = min(box_w, img_w - x_min)
            box_h = min(box_h, img_h - y_min)

            x_center = (x_min + box_w / 2) / img_w
            y_center = (y_min + box_h / 2) / img_h
            norm_w   = box_w / img_w
            norm_h   = box_h / img_h

            # COCO category_id starts at 1; YOLO class_id starts at 0
            class_id = ann["category_id"] - 1

            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

        label_path = labels_dir / f"{file_stem}.txt"

        if lines:
            label_path.write_text("\n".join(lines) + "\n")
            converted += 1
        else:
            label_path.touch()   # empty file = image with no objects (background)
            empty += 1

    print(f"  -> {converted} label files written  ({empty} images with no annotations)\n")


def main() -> None:
    print("=" * 50)
    print("  COCO JSON -> YOLO TXT Annotation Converter")
    print("=" * 50 + "\n")

    for split in SPLITS:
        convert_split(split)

    print("=" * 50)
    print("  Done! Now run: python training/train_model.py")
    print("=" * 50)


if __name__ == "__main__":
    main()