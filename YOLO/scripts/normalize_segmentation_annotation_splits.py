"""OBB pseudo annotation listesindeki kaynak splitlerini normalize eder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    path = args.annotations.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = {"train": "train", "valid": "val", "test": "test"}
    for item in data:
        source_image = str(item.get("source_image", ""))
        source_split = source_image.split("/")[2] if "/" in source_image else "train"
        item["split"] = mapping.get(source_split, item.get("split", "train"))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Splitleri normalize edildi: {len(data)} kayıt")


if __name__ == "__main__":
    main()
