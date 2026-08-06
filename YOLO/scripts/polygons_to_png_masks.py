"""Polygon segmentasyon etiketlerini tek kanallı PNG maskelere çevirir.

Girdi JSON dizisi örneği:

[
  {
    "image": "raw/phone-validation/images/capture-index.jpg",
    "mask": "masks/phone-validation/capture-index.png",
    "polygons": [[[12, 30], [40, 15], [55, 90]]],
    "coordinate_space": "pixel",
    "person_id": "phone-validation",
    "split": "val",
    "finger_position": "index"
  }
]

Polygonlar görüntü koordinatlarında tutulur; çıktı maskesinde 255 foreground,
0 background anlamına gelir. Boş polygonlar varsayılan olarak hata üretir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def load_annotation_files(paths: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Annotation JSON dizisi değil: {path}")
        entries.extend(data)
    return entries


def normalized_points(
    polygon: list[list[float]], width: int, height: int, coordinate_space: str
) -> list[tuple[float, float]]:
    if coordinate_space == "normalized":
        return [(x * (width - 1), y * (height - 1)) for x, y in polygon]
    return [(float(x), float(y)) for x, y in polygon]


def create_mask(entry: dict[str, Any], root: Path, allow_empty: bool) -> dict[str, Any]:
    image_path = (root / entry["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")
    if "mask" not in entry:
        raise ValueError(f"Mask yolu eksik: {image_path}")

    with Image.open(image_path) as image:
        width, height = image.size
    polygons = entry.get("polygons", [])
    if not isinstance(polygons, list):
        raise ValueError(f"Polygon alanı dizi değil: {image_path}")
    if not polygons and not allow_empty:
        raise ValueError(
            f"Polygon boş: {image_path}. Elle etiketleyin veya --allow-empty kullanın."
        )

    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)
    coordinate_space = entry.get("coordinate_space", "pixel")
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError(f"Polygon en az üç nokta içermeli: {image_path}")
        points = normalized_points(polygon, width, height, coordinate_space)
        drawer.polygon(points, fill=255)

    mask_path = (root / entry["mask"]).resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path, format="PNG", optimize=True)
    output = dict(entry)
    output["width"] = width
    output["height"] = height
    output["mask"] = mask_path.relative_to(root).as_posix()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Polygonları PNG maskeye çevirir.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Boş polygonlar için siyah maske üretir; eğitim öncesi önerilmez.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    entries = load_annotation_files([path.resolve() for path in args.annotations])
    if args.max_items is not None:
        entries = entries[: max(0, args.max_items)]
    converted = []
    skipped = 0
    for entry in entries:
        if not entry.get("polygons") and args.allow_empty:
            skipped += 1
        converted.append(create_mask(entry, root, args.allow_empty))

    output_path = args.output_annotations.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{len(converted)} maske yazıldı: {output_path}")
    if skipped:
        print(f"Uyarı: {skipped} kayıt boş polygon ile siyah maske aldı.")


if __name__ == "__main__":
    main()
