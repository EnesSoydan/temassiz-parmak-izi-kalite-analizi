"""OBB modelini dışarıdan verilen veri setiyle doğrular."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "fingertip_obb"
    / "brightness_distance_obb"
    / "weights"
    / "best.pt"
)
DEFAULT_DATA = (
    PROJECT_ROOT
    / "YOLO"
    / "datasets"
    / "merged_hands_fingertip_obb"
    / "data.yaml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "runs"
    / "fingertip_obb"
    / "brightness_distance_obb"
    / "val_results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO OBB validation")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Örnek: 0 veya cpu")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(
            f"data.yaml bulunamadı: {data_path}. Veri yolunu --data ile belirtin."
        )

    model = YOLO(str(model_path), task="obb")
    model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        iou=0.65,
        conf=0.5,
        task="val",
        device=args.device,
        workers=args.workers,
        save=True,
        save_txt=True,
        save_conf=True,
        save_json=True,
        plots=True,
        verbose=True,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    validate(parse_args())
