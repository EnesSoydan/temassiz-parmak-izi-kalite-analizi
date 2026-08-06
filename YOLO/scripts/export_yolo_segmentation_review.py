"""Pseudo polygon annotations'ı YOLO segmentation inceleme paketine dönüştürür.

Üretilen yapı Ultralytics/YOLO segmentation araçlarında açılabilir:

    output/
      data.yaml
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt

Her label satırı şu biçimdedir:

    class_id x1 y1 x2 y2 ...

Koordinatlar görüntü boyutuna göre 0..1 aralığına normalize edilir. Kaynak
annotation içindeki polygonlar piksel koordinatında değilse `coordinate_space`
alanı üzerinden ayrıca normalize edilir.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


CLASS_NAMES = ("index", "middle", "ring", "pinky")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def polygon_to_yolo(
    polygon: list[list[float]],
    width: int,
    height: int,
    coordinate_space: str,
) -> list[float]:
    if len(polygon) < 3:
        raise ValueError("Polygon en az üç nokta içermeli.")

    normalized: list[float] = []
    for point in polygon:
        if len(point) != 2:
            raise ValueError(f"Geçersiz polygon noktası: {point!r}")
        x, y = float(point[0]), float(point[1])
        if coordinate_space == "normalized":
            nx, ny = x, y
        elif coordinate_space == "pixel":
            nx, ny = x / max(width, 1), y / max(height, 1)
        else:
            raise ValueError(f"Desteklenmeyen coordinate_space: {coordinate_space!r}")
        normalized.extend(
            (
                clamp(nx, 0.0, 1.0),
                clamp(ny, 0.0, 1.0),
            )
        )
    return normalized


def write_data_yaml(
    output_root: Path, include_test: bool, class_names: tuple[str, ...]
) -> None:
    data_yaml = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        data_yaml.append("test: images/test")
    data_yaml.append("names:")
    data_yaml.extend(f"  {class_id}: {name}" for class_id, name in enumerate(class_names))
    data_yaml.append("")
    (output_root / "data.yaml").write_text("\n".join(data_yaml), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polygon JSON'ını YOLO segmentation inceleme paketine dönüştürür."
    )
    parser.add_argument("--root", type=Path, required=True, help="Annotation yollarının kökü")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Test splitini de dışa aktar. İnceleme için önerilir.",
    )
    parser.add_argument(
        "--class-mode",
        choices=("single", "finger-position"),
        default="single",
        help="Segmentasyon sınıfı: single=tek finger sınıfı, finger-position=4 parmak sınıfı.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="İsteğe bağlı smoke-test sınırı")
    args = parser.parse_args()

    root = args.root.resolve()
    annotations_path = args.annotations.resolve()
    output_root = args.output.resolve()
    entries = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Annotation JSON boş veya dizi değil.")
    if args.max_items is not None:
        entries = entries[: args.max_items]

    class_names = ("finger",) if args.class_mode == "single" else CLASS_NAMES

    exported: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for entry in entries:
        split = str(entry.get("split", "train"))
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Geçersiz split: {split!r}")
        if split == "test" and not args.include_test:
            continue

        finger_position = str(entry.get("finger_position", "")).lower()
        if finger_position not in CLASS_NAMES:
            raise ValueError(f"Geçersiz finger_position: {finger_position!r}")
        class_id = 0 if args.class_mode == "single" else CLASS_NAMES.index(finger_position)

        image_path = (root / str(entry["image"])).resolve()
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")
        polygons = entry.get("polygons")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"Polygon bulunamadı: {entry.get('image')}")

        with Image.open(image_path) as image:
            width, height = image.size

        coordinate_space = str(entry.get("coordinate_space", "pixel"))
        label_lines: list[str] = []
        for polygon in polygons:
            normalized = polygon_to_yolo(polygon, width, height, coordinate_space)
            label_lines.append(
                str(class_id)
                + " "
                + " ".join(f"{value:.6f}" for value in normalized)
            )

        # Split klasörleri korunur; aynı capture/detection adları farklı splitlerde
        # çakışsa bile birbirlerinin üstüne yazmaz.
        image_output = output_root / "images" / split / image_path.name
        label_output = output_root / "labels" / split / f"{image_path.stem}.txt"
        image_output.parent.mkdir(parents=True, exist_ok=True)
        label_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_output)
        label_output.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        exported.append(
            {
                "image": image_output.relative_to(output_root).as_posix(),
                "label": label_output.relative_to(output_root).as_posix(),
                "source_image": entry.get("source_image", entry["image"]),
                "finger_position": finger_position,
                "class_id": class_id,
                "confidence": entry.get("confidence"),
                "label_status": entry.get("label_status", "unknown"),
            }
        )
        split_counts[split] = split_counts.get(split, 0) + 1

    if not exported:
        raise ValueError("Dışa aktarılacak annotation bulunamadı.")

    output_root.mkdir(parents=True, exist_ok=True)
    write_data_yaml(output_root, include_test=args.include_test, class_names=class_names)
    (output_root / "review-metadata.json").write_text(
        json.dumps(
            {
                "format": "yolo-segmentation",
                "class_mode": args.class_mode,
                "class_names": list(class_names),
                "source_annotations": str(annotations_path),
                "entries": exported,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"YOLO segmentation paketi yazıldı: {output_root} "
        f"({len(exported)} örnek, "
        + ", ".join(f"{key}={value}" for key, value in sorted(split_counts.items()))
        + ")"
    )


if __name__ == "__main__":
    main()
