"""PNG maskeler ve kişi bazlı metadata'dan U-Net manifesti üretir.

Annotations JSON dizisi örneği:
[
    {"image": "images/sample-index.jpg", "mask": "masks/sample-index.png", "person_id": "p01"}
]

Kişi split'i burada yapıldığı için aynı kişinin tekrar çekimleri train ve val'e
karışmaz. Üretilen manifest Git'e alınmamalıdır.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-people-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--respect-explicit-split",
        action="store_true",
        help="Annotation içindeki train/val split değerlerini korur.",
    )
    parser.add_argument(
        "--exclude-split",
        nargs="*",
        choices=["train", "val", "test"],
        default=[],
        help="Manifestten çıkarılacak splitler; örneğin test.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("Annotations JSON boş veya dizi değil.")
    annotations = [
        item for item in annotations if item.get("split") not in set(args.exclude_split)
    ]
    if not annotations:
        raise ValueError("Seçilen splitler çıkarılınca annotation kalmadı.")

    if args.respect_explicit_split:
        invalid = [
            item.get("split")
            for item in annotations
            if item.get("split") not in {"train", "val", "test"}
        ]
        if invalid:
            raise ValueError("Explicit split kullanılırken her kayıt train, val veya test olmalı.")
        people_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        for item in annotations:
            people_by_split[item["split"]].add(str(item["person_id"]))
        overlap = people_by_split["train"] & people_by_split["val"]
        if overlap:
            raise ValueError(
                "Aynı kişi train ve val kümelerinde bulunuyor: "
                + ", ".join(sorted(overlap))
            )
        test_overlap = people_by_split["test"] & (
            people_by_split["train"] | people_by_split["val"]
        )
        if test_overlap:
            raise ValueError(
                "Aynı kişi train/val ve test kümelerinde bulunuyor: "
                + ", ".join(sorted(test_overlap))
            )
        val_people = people_by_split["val"]
    else:
        people = sorted({str(item["person_id"]) for item in annotations})
        random.Random(args.seed).shuffle(people)
        val_count = max(1, round(len(people) * args.val_people_ratio)) if len(people) > 1 else 0
        val_people = set(people[:val_count])
    entries = []
    for item in annotations:
        image = (root / item["image"]).resolve()
        mask = (root / item["mask"]).resolve()
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Görsel veya maske bulunamadı: {image}, {mask}")
        person_id = str(item["person_id"])
        split = item.get("split") if args.respect_explicit_split else None
        entries.append(
            {
                "image": image.relative_to(root).as_posix(),
                "mask": mask.relative_to(root).as_posix(),
                "person_id": person_id,
                "split": split or ("val" if person_id in val_people else "train"),
                "finger_position": item.get("finger_position", "unknown"),
                "capture_id": item.get("capture_id", image.stem),
                "source": item.get("source", "unknown"),
                "label_status": item.get("label_status", "reviewed"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest yazıldı: {args.output} ({len(entries)} örnek, val kişi={len(val_people)})")


if __name__ == "__main__":
    main()
