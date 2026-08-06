"""Tek sınıflı parmak segmentation modeli eğitimi.

Varsayılan veri seti, SAM2 ile otomatik polygon üretilmiş ve 80/20 ayrılmış
1500 örneklik inceleme/eğitim paketidir:

    YOLO/segmentation_data/yolo-sam2-review-1500/data.yaml

Bu script mevcut OBB `training.py` dosyasından ayrıdır. Böylece OBB eğitim
akışı değişmeden kalır.

Örnek:
    python YOLO/scripts/training_segmentation.py

GPU ve ONNX export ile:
    python YOLO/scripts/training_segmentation.py --device 0 --export-onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    PROJECT_ROOT
    / "YOLO"
    / "segmentation_data"
    / "yolo-sam2-review-1500"
    / "data.yaml"
)
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "fingertip_segmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM2 polygonlarıyla tek sınıflı YOLO segmentation eğitimi."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="YOLO segmentation data.yaml dosyası.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n-seg.pt",
        help="Başlangıç segmentation modeli (.pt veya Ultralytics model adı).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=480,
        help="Kanonik ROI ve mobil uygulama giriş boyutuyla eşleşir.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="Örnek: 0 veya cpu")
    parser.add_argument(
        "--name",
        default="sam2_review_1500_yolo11n_seg",
        help="Eğitim koşusu adı.",
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Eğitim sonrası best.pt dosyasını ONNX olarak dışa aktarır.",
    )
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Validation gelişmezse erken durdurma sabrı.",
    )
    return parser.parse_args()


def resolve_data_yaml(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Segmentation data.yaml bulunamadı: {resolved}")
    return resolved


def validate_dataset_config(data_yaml: Path) -> dict:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("data.yaml sözlük biçiminde değil.")
    names = config.get("names")
    if isinstance(names, dict):
        class_names = [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    elif isinstance(names, list):
        class_names = [str(value) for value in names]
    else:
        raise ValueError("data.yaml içinde names bulunamadı.")
    if class_names != ["finger"]:
        raise ValueError(
            f"Bu eğitim tek sınıf bekliyor; mevcut sınıflar: {class_names!r}"
        )
    if "train" not in config or "val" not in config:
        raise ValueError("data.yaml içinde train ve val yolları bulunmalı.")

    dataset_root_value = config.get("path", data_yaml.parent)
    dataset_root = Path(str(dataset_root_value)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()

    for key in ("train", "val"):
        split_path = Path(str(config[key]))
        split_path = split_path if split_path.is_absolute() else dataset_root / split_path
        if not split_path.is_dir():
            raise FileNotFoundError(f"{key} görüntü klasörü bulunamadı: {split_path.resolve()}")

    return config


def select_device(requested: str | None) -> str:
    if requested:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def train(args: argparse.Namespace) -> Path:
    data_yaml = resolve_data_yaml(args.data)
    validate_dataset_config(data_yaml)
    device = select_device(args.device)

    print(f"Model: {args.model}")
    print(f"Veri: {data_yaml}")
    print(f"Cihaz: {device}")
    print(f"Görüntü boyutu: {args.imgsz}x{args.imgsz}")
    print(f"Epoch: {args.epochs}, batch: {args.batch}")

    model = YOLO(args.model)
    model.train(
        task="segment",
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=max(10, args.epochs // 8),
        amp=True,
        cache=args.cache,
        plots=True,
        save=True,
        save_period=10,
        seed=42,
        deterministic=True,
        # Kanonik ROI zaten hizalı olduğu için aşırı geometrik dönüşüm kullanmıyoruz.
        degrees=10.0,
        translate=0.06,
        scale=0.20,
        shear=4.0,
        perspective=0.0003,
        fliplr=0.5,
        flipud=0.0,
        # Hafif fotometrik çeşitlilik; SAM2 maskeleriyle geometrik eşleşme korunur.
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.35,
        mosaic=0.15,
        mixup=0.0,
        copy_paste=0.0,
        val=True,
    )

    trainer = model.trainer
    if trainer is None or not trainer.best:
        raise RuntimeError("Eğitim tamamlandı ancak best.pt yolu bulunamadı.")
    best_path = Path(trainer.best).resolve()
    if not best_path.is_file():
        raise FileNotFoundError(f"best.pt bulunamadı: {best_path}")

    print(f"Best model: {best_path}")
    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        plots=True,
    )
    print(f"Validation mAP50: {metrics.seg.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.seg.map:.4f}")

    if args.export_onnx:
        exported = best_model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=True,
            dynamic=False,
        )
        print(f"ONNX model: {exported}")

    return best_path


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
