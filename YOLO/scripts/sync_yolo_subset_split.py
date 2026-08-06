"""Seçilmiş YOLO görsellerinin label eşlerini bulur ve 80/20 böler.

Subset klasörüne önceden kopyalanmış görsellerin aynı isimli label dosyalarını
ana YOLO segmentation datasetinden bulur. Split, mümkün olduğunda aynı
capture içindeki dört parmağı train ve valid arasında bölmemek için capture
grubuna göre yapılır.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FINGER_SUFFIX = re.compile(r"^(?P<capture>.+)-det\d{2}-(?:index|middle|ring|pinky)$", re.IGNORECASE)


def capture_group(stem: str) -> str:
    match = FINGER_SUFFIX.match(stem)
    return match.group("capture") if match else stem


def find_image_root(target: Path) -> Path:
    candidates = [target / "train" / "images", target / "images"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Subset görsel klasörü bulunamadı: {target}")


def build_label_index(source: Path) -> dict[str, Path]:
    labels = list((source / "labels").rglob("*.txt"))
    index: dict[str, Path] = {}
    for label in labels:
        key = label.name.lower()
        if key in index:
            raise ValueError(f"Aynı isimli birden fazla kaynak label bulundu: {label.name}")
        index[key] = label
    return index


def write_data_yaml(target: Path) -> None:
    data_yaml = "\n".join(
        [
            f"path: {target.as_posix()}",
            "train: train/images",
            "val: valid/images",
            "names:",
            "  0: finger",
            "",
        ]
    )
    (target / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO subset label eşleştirme ve train/valid split")
    parser.add_argument("--source", type=Path, required=True, help="Tam YOLO dataset kökü")
    parser.add_argument("--target", type=Path, required=True, help="Önceden görselleri kopyalanmış subset")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train-ratio 0 ile 1 arasında olmalı.")
    if not source.is_dir() or not target.is_dir():
        raise FileNotFoundError(f"Source veya target klasörü bulunamadı: {source}, {target}")

    image_root = find_image_root(target)
    images = sorted(
        path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"Subset içinde görsel bulunamadı: {image_root}")
    if len({path.name.lower() for path in images}) != len(images):
        raise ValueError("Subset içinde aynı isimli birden fazla görsel var; güvenli eşleştirme yapılamadı.")

    label_index = build_label_index(source)
    missing = [path.name for path in images if f"{path.stem}.txt".lower() not in label_index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} görselin source datasetinde label karşılığı yok. İlk örnek: {missing[:5]}"
        )

    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[capture_group(image.stem)].append(image)
    group_list = list(groups.values())
    random.Random(args.seed).shuffle(group_list)
    target_train = round(len(images) * args.train_ratio)
    train_groups: list[list[Path]] = []
    valid_groups: list[list[Path]] = []
    train_count = 0
    for group in group_list:
        # Grubu bölmemek için hedefe en yakın dağılımı seç.
        if train_count < target_train:
            train_groups.append(group)
            train_count += len(group)
        else:
            valid_groups.append(group)

    # Çok büyük/tek gruplu küçük subsetlerde en az bir valid örneği garantile.
    if not valid_groups and len(train_groups) > 1:
        valid_groups.append(train_groups.pop())
    train_images = [image for group in train_groups for image in group]
    valid_images = [image for group in valid_groups for image in group]
    if not train_images or not valid_images:
        raise ValueError("Train veya valid split boş kaldı.")

    train_image_root = target / "train" / "images"
    valid_image_root = target / "valid" / "images"
    train_label_root = target / "train" / "labels"
    valid_label_root = target / "valid" / "labels"
    for directory in (train_image_root, valid_image_root, train_label_root, valid_label_root):
        directory.mkdir(parents=True, exist_ok=True)

    valid_set = {image.name.lower() for image in valid_images}
    for image in images:
        split = "valid" if image.name.lower() in valid_set else "train"
        image_destination = target / split / "images" / image.name
        if image.resolve() != image_destination.resolve():
            shutil.move(str(image), str(image_destination))
        label_source = label_index[f"{image.stem}.txt".lower()]
        label_destination = target / split / "labels" / f"{image.stem}.txt"
        shutil.copy2(label_source, label_destination)

    write_data_yaml(target)
    print(
        f"Subset hazırlandı: {target} "
        f"(train={len(train_images)}, valid={len(valid_images)}, toplam={len(images)}, "
        f"train_oranı={len(train_images) / len(images):.3f})"
    )


if __name__ == "__main__":
    main()
