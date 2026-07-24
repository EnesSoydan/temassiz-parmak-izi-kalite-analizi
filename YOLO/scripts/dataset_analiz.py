from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_ROOT = Path(r"C:\Users\eness\Downloads\final_hands")
DEFAULT_SPLITS = ("train", "valid", "test")


# data.yaml icindeki names alanini basit YOLO formatlarindan okumaya calisir.
def read_class_names(dataset_root: Path) -> dict[int, str]:
    data_yaml = dataset_root / "data.yaml"

    if not data_yaml.exists():
        return {}

    lines = data_yaml.read_text(encoding="utf-8", errors="ignore").splitlines()
    class_names: dict[int, str] = {}

    for index, line in enumerate(lines):
        stripped = line.strip()

        # Ornek: names: ['hand', 'index']
        if stripped.startswith("names:") and "[" in stripped and "]" in stripped:
            raw_names = stripped.split(":", 1)[1].strip().strip("[]")
            for class_id, name in enumerate(raw_names.split(",")):
                cleaned = name.strip().strip("'\"")
                if cleaned:
                    class_names[class_id] = cleaned
            return class_names

        # Ornek:
        # names:
        #   0: hand
        if stripped == "names:":
            for next_line in lines[index + 1 :]:
                if not next_line.startswith((" ", "\t")):
                    break

                item = next_line.strip()
                if ":" not in item:
                    continue

                key, value = item.split(":", 1)
                if key.strip().isdigit():
                    class_names[int(key.strip())] = value.strip().strip("'\"")

            return class_names

    return class_names


# Label satirinin bbox, OBB veya baska bir YOLO turevi olup olmadigini kaba olarak siniflandirir.
def detect_label_format(parts: list[str]) -> str:
    if len(parts) == 5:
        return "bbox"

    if len(parts) == 9:
        return "obb"

    if len(parts) > 5 and len(parts[1:]) % 2 == 0:
        return "polygon"

    return "unknown"


# Tek bir split icin gorsel, label ve sinif dagilimi istatistiklerini hesaplar.
def analyze_split(dataset_root: Path, split: str) -> dict:
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"

    stats = {
        "images": 0,
        "label_files": 0,
        "nonempty_label_files": 0,
        "empty_label_files": 0,
        "missing_label_files": 0,
        "objects": 0,
        "bad_lines": 0,
        "class_counts": Counter(),
        "format_counts": Counter(),
        "bad_examples": [],
    }

    if not images_dir.exists():
        stats["missing_split"] = True
        return stats

    images = sorted(
        image
        for image in images_dir.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    )
    stats["images"] = len(images)

    for image in images:
        label_path = labels_dir / f"{image.stem}.txt"

        if not label_path.exists():
            stats["missing_label_files"] += 1
            continue

        stats["label_files"] += 1
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]

        if not lines:
            stats["empty_label_files"] += 1
            continue

        stats["nonempty_label_files"] += 1

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()
            label_format = detect_label_format(parts)
            stats["format_counts"][label_format] += 1

            if len(parts) < 2 or not parts[0].isdigit() or label_format == "unknown":
                stats["bad_lines"] += 1
                remember_bad_example(stats, label_path, line_number, line)
                continue

            try:
                values = [float(value) for value in parts[1:]]
            except ValueError:
                stats["bad_lines"] += 1
                remember_bad_example(stats, label_path, line_number, line)
                continue

            # YOLO koordinatlari normalize oldugu icin 0-1 araligindan tasanlari bozuk satir sayariz.
            if any(value < 0 or value > 1 for value in values):
                stats["bad_lines"] += 1
                remember_bad_example(stats, label_path, line_number, line)
                continue

            class_id = int(parts[0])
            stats["objects"] += 1
            stats["class_counts"][class_id] += 1

    stats["missing_split"] = False
    return stats


# Raporu sisirmemek icin ilk birkac bozuk satir ornegini saklar.
def remember_bad_example(stats: dict, label_path: Path, line_number: int, line: str) -> None:
    if len(stats["bad_examples"]) >= 5:
        return

    stats["bad_examples"].append(f"{label_path}:{line_number} -> {line}")


# Sinif id'lerini data.yaml isimleriyle birlikte okunur sekilde yazar.
def format_class_counts(class_counts: Counter, class_names: dict[int, str]) -> list[str]:
    rows = []

    for class_id, count in sorted(class_counts.items()):
        name = class_names.get(class_id, f"class_{class_id}")
        rows.append(f"    {class_id} ({name}): {count}")

    return rows or ["    yok"]


# Analiz sonucunu terminalde hizli okunacak sekilde basar.
def print_report(dataset_root: Path, splits: tuple[str, ...]) -> None:
    class_names = read_class_names(dataset_root)
    totals = defaultdict(int)
    total_class_counts = Counter()
    total_format_counts = Counter()

    print(f"Dataset: {dataset_root}")
    print()

    for split in splits:
        stats = analyze_split(dataset_root, split)

        if stats.get("missing_split"):
            print(f"[{split}] images klasoru bulunamadi, atlandi.")
            print()
            continue

        totals["images"] += stats["images"]
        totals["label_files"] += stats["label_files"]
        totals["nonempty_label_files"] += stats["nonempty_label_files"]
        totals["empty_label_files"] += stats["empty_label_files"]
        totals["missing_label_files"] += stats["missing_label_files"]
        totals["objects"] += stats["objects"]
        totals["bad_lines"] += stats["bad_lines"]
        total_class_counts.update(stats["class_counts"])
        total_format_counts.update(stats["format_counts"])

        print(f"[{split}]")
        print(f"  gorsel sayisi: {stats['images']}")
        print(f"  etiketli gorsel: {stats['nonempty_label_files']}")
        print(f"  label dosyasi olan gorsel: {stats['label_files']}")
        print(f"  dolu label dosyasi: {stats['nonempty_label_files']}")
        print(f"  bos label dosyasi: {stats['empty_label_files']}")
        print(f"  label dosyasi olmayan gorsel: {stats['missing_label_files']}")
        print(f"  obje etiketi: {stats['objects']}")
        print(f"  bozuk satir: {stats['bad_lines']}")
        print("  format dagilimi:")
        for label_format, count in sorted(stats["format_counts"].items()):
            print(f"    {label_format}: {count}")
        print("  sinif dagilimi:")
        print("\n".join(format_class_counts(stats["class_counts"], class_names)))

        if stats["bad_examples"]:
            print("  ilk bozuk satir ornekleri:")
            for example in stats["bad_examples"]:
                print(f"    {example}")

        print()

    print("[toplam]")
    print(f"  gorsel sayisi: {totals['images']}")
    print(f"  etiketli gorsel: {totals['nonempty_label_files']}")
    print(f"  label dosyasi olan gorsel: {totals['label_files']}")
    print(f"  dolu label dosyasi: {totals['nonempty_label_files']}")
    print(f"  bos label dosyasi: {totals['empty_label_files']}")
    print(f"  label dosyasi olmayan gorsel: {totals['missing_label_files']}")
    print(f"  obje etiketi: {totals['objects']}")
    print(f"  bozuk satir: {totals['bad_lines']}")
    print("  format dagilimi:")
    for label_format, count in sorted(total_format_counts.items()):
        print(f"    {label_format}: {count}")
    print("  sinif dagilimi:")
    print("\n".join(format_class_counts(total_class_counts, class_names)))


# Komut satirindan dataset yolu ve split listesi alir.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO dataset etiketlilik analizi yapar.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(DEFAULT_DATASET_ROOT),
        help=f"Dataset kok klasoru. Varsayilan: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Analiz edilecek split isimleri. Ornek: --splits train valid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    print_report(dataset_root, tuple(args.splits))


if __name__ == "__main__":
    main()
