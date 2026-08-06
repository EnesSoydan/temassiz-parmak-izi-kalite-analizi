"""Segmentasyon etiketleme çalışması için yalnızca görsel kaynaklarını hazırlar.

OBB veri setinin ``labels`` klasörleri bilinçli olarak kopyalanmaz. Böylece bu
klasör, daha sonra gerçek veya elle düzeltilmiş segmentasyon etiketleriyle
ayrı bir çalışma alanı olarak kullanılabilir.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def copy_split(source_root: Path, output_root: Path, split: str) -> list[dict]:
    source_images = source_root / split / "images"
    if not source_images.is_dir():
        raise FileNotFoundError(f"Görsel klasörü bulunamadı: {source_images}")

    destination = output_root / "raw" / "obb" / split / "images"
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for source in sorted(source_images.iterdir()):
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        entries.append(
            {
                "image": target.relative_to(output_root).as_posix(),
                "source_split": split,
                "source_path": str(source),
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OBB veri setinden etiketleri almadan görselleri kopyalar."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    all_entries: list[dict] = []
    for split in args.splits:
        entries = copy_split(source_root, output_root, split)
        all_entries.extend(entries)
        print(f"{split}: {len(entries)} görsel kopyalandı")

    metadata_path = output_root / "annotations" / "obb-source-images.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(all_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Toplam: {len(all_entries)} görsel")
    print(f"Kaynak listesi: {metadata_path}")


if __name__ == "__main__":
    main()
