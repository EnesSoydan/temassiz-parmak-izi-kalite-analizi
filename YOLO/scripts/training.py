"""Parlaklık ve uzaklık değişimlerine dayanıklı YOLO OBB eğitimi.

Yerel kullanım örneği:
python YOLO/scripts/training.py --data "C:/veri/data.yaml"

Yolo-Training-and-Inference arayüzü:
Script dosyasını seçip doğrudan başlat; varsayılan veri seti ve eğitim ayarları kullanılır.

Colab kullanım örneği:
python YOLO/scripts/training.py \
    --data "/content/drive/MyDrive/merged_hands_fingertip_obb/data.yaml" \
    --model "/content/drive/MyDrive/best.pt"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "final_hands_yolo11n_obb" / "weights" / "best.pt"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "fingertip_obb"
IMAGE_SPLIT_NAMES = ("train", "val", "valid")
DATA_YAML = PROJECT_ROOT / "YOLO" / "datasets" / "merged_hands_fingertip_obb" / "data.yaml"


# Komut satırından veri seti, başlangıç modeli ve temel eğitim ayarlarını alır.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parlaklık ve uzaklık augmentasyonlarıyla YOLO OBB modeli eğitir."
    )
    # Masaüstü eğitim arayüzü argüman göndermediği için proje içindeki data.yaml varsayılan kullanılır.
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_YAML,
        help="YOLO data.yaml dosyası.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL if DEFAULT_MODEL.exists() else "yolo11n-obb.pt"),
        help="Başlangıç .pt modeli veya Ultralytics model adı.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Eğitim epoch sayısı.")
    parser.add_argument("--batch", type=int, default=12, help="Batch boyutu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Eğitim görüntü boyutu.")
    parser.add_argument(
        "--mobile-imgsz",
        type=int,
        default=320,
        help="Mobil validation hedef görüntü boyutu.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Veri yükleme worker sayısı.")
    parser.add_argument("--device", default=None, help="Örnek: 0, 0,1 veya cpu.")
    parser.add_argument("--name", default="brightness_distance_obb", help="Eğitim koşusu adı.")
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Eğitim sonuçlarının yazılacağı klasör.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Veri setini önbelleğe alır; yeterli RAM varsa kullanılmalı.",
    )
    parser.add_argument(
        "--skip-mobile-val",
        action="store_true",
        help="Eğitim sonundaki 320x320 mobil validation adımını atlar.",
    )
    return parser.parse_args()


# data.yaml içeriğini okuyup eğitim başlamadan önce temel dosya hatalarını yakalar.
def load_dataset_config(data_yaml: Path) -> dict:
    resolved_yaml = data_yaml.expanduser().resolve()

    # Yanlış yol verilirse Ultralytics'in uzun hata mesajından önce doğrudan sebebi gösteririz.
    if not resolved_yaml.is_file():
        raise FileNotFoundError(f"data.yaml bulunamadı: {resolved_yaml}")

    config = yaml.safe_load(resolved_yaml.read_text(encoding="utf-8"))

    # Eğitim için sınıf isimleri ve train yolu mutlaka bulunmalıdır.
    if not isinstance(config, dict) or "train" not in config or "names" not in config:
        raise ValueError("data.yaml içinde en az train ve names alanları bulunmalı.")

    return config


# data.yaml içindeki path alanını yaml konumuna göre mutlak veri seti yoluna çevirir.
def resolve_dataset_root(data_yaml: Path, config: dict) -> Path:
    configured_root = config.get("path")

    # path verilmemişse train/valid klasörlerinin data.yaml yanında olduğunu kabul ederiz.
    if configured_root in (None, ""):
        return data_yaml.expanduser().resolve().parent

    root = Path(str(configured_root)).expanduser()

    # Göreli path değerleri data.yaml dosyasının bulunduğu klasöre göre çözülür.
    if not root.is_absolute():
        root = data_yaml.expanduser().resolve().parent / root

    return root.resolve()


# Görsel split yolundan aynı yapıda bulunan labels klasörünü bulur.
def resolve_labels_directory(dataset_root: Path, split_value: str) -> Path | None:
    split_path = Path(split_value)

    # URL veya Ultralytics'in uzak veri kaynakları yerel etiket analizine uygun değildir.
    if "://" in split_value:
        return None

    image_path = split_path if split_path.is_absolute() else dataset_root / split_path
    parts = list(image_path.parts)

    # Standart split/images yapısını split/labels biçimine dönüştürürüz.
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts)

    return image_path.parent / "labels"


# Normalize OBB köşe noktalarından görüntü alanına oranlı poligon alanını hesaplar.
def calculate_obb_area(values: list[float]) -> float:
    points = [(values[index], values[index + 1]) for index in range(0, 8, 2)]
    doubled_area = 0.0

    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        doubled_area += point[0] * next_point[1] - next_point[0] * point[1]

    return abs(doubled_area) / 2


# Eğitim etiketlerindeki küçük parmak ucu oranını raporlayarak uzaklık veri açığını görünür yapar.
def report_object_scale_distribution(data_yaml: Path, config: dict) -> None:
    dataset_root = resolve_dataset_root(data_yaml, config)
    train_value = config.get("train")

    # Birden fazla train kaynağı varsa ilk yerel kaynağı analiz ederiz.
    if isinstance(train_value, list):
        train_value = next((value for value in train_value if isinstance(value, str)), None)

    if not isinstance(train_value, str):
        print("[ölçek analizi] Train yolu çözülemedi; etiket alanı analizi atlandı.")
        return

    labels_dir = resolve_labels_directory(dataset_root, train_value)

    # data.yaml başka bir makinenin mutlak yolunu taşıyorsa eğitimi Ultralytics'e bırakıp uyarı veririz.
    if labels_dir is None or not labels_dir.is_dir():
        print(f"[ölçek analizi] Labels klasörü bulunamadı: {labels_dir}")
        print("[ölçek analizi] data.yaml içindeki path/train yollarını kontrol et.")
        return

    areas: list[float] = []
    invalid_lines = 0

    for label_path in labels_dir.rglob("*.txt"):
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()

            # OBB etiketi class_id ve sekiz normalize koordinattan oluşmalıdır.
            if len(parts) != 9:
                if parts:
                    invalid_lines += 1
                continue

            try:
                coordinates = [float(value) for value in parts[1:]]
            except ValueError:
                invalid_lines += 1
                continue

            areas.append(calculate_obb_area(coordinates))

    # Dolu OBB etiketi yoksa eğitimden önce açık bir uyarı üretiriz.
    if not areas:
        print("[ölçek analizi] Geçerli OBB etiketi bulunamadı.")
        return

    very_small_count = sum(area < 0.005 for area in areas)
    small_count = sum(area < 0.015 for area in areas)
    very_small_ratio = very_small_count / len(areas)
    small_ratio = small_count / len(areas)

    print("[ölçek analizi]")
    print(f"  toplam OBB etiketi: {len(areas)}")
    print(f"  medyan kutu alanı: %{median(areas) * 100:.3f}")
    print(f"  çok küçük kutu (<%0.5): {very_small_count} (%{very_small_ratio * 100:.1f})")
    print(f"  küçük kutu (<%1.5): {small_count} (%{small_ratio * 100:.1f})")
    print(f"  bozuk/OBB olmayan satır: {invalid_lines}")

    # Uzak çekim başarısı için augmentasyonun yanında gerçek küçük el örnekleri de bulunmalıdır.
    if small_ratio < 0.15:
        print(
            "  UYARI: Küçük parmak ucu oranı düşük. "
            "Veri setine gerçekten uzaktan çekilmiş ve elle doğrulanmış örnekler ekle."
        )


# CUDA varsa ilk GPU'yu, yoksa CPU'yu seçer; kullanıcı --device ile bunu değiştirebilir.
def select_device(requested_device: str | None) -> str:
    if requested_device:
        return requested_device

    return "0" if torch.cuda.is_available() else "cpu"


# Parlaklık ve nesne ölçeği çeşitliliğini artıran OBB eğitimini başlatır.
def train_model(args: argparse.Namespace) -> YOLO:
    data_yaml = args.data.expanduser().resolve()
    config = load_dataset_config(data_yaml)
    report_object_scale_distribution(data_yaml, config)

    print()
    print(f"Başlangıç modeli: {args.model}")
    print(f"Veri seti: {data_yaml}")
    print(f"Cihaz: {select_device(args.device)}")

    model = YOLO(args.model)
    model.train(
        task="obb",
        data=str(data_yaml),
        epochs=args.epochs,
        patience=15,
        batch=args.batch,
        imgsz=args.imgsz,
        device=select_device(args.device),
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
        # Farklı ortam ışıklarını ve telefon pozlamalarını taklit eder.
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.60,
        # Uzak/yakın el ölçeğini ve kadraj içindeki konum değişimini taklit eder.
        scale=0.70,
        translate=0.15,
        multi_scale=0.25,
        # Düşük oranlı mosaic küçük nesne görünümünü artırır; son epochlarda otomatik kapanır.
        mosaic=0.35,
        mixup=0.0,
        # OBB modelinin telefon ve el açısı değişimlerine toleransını artırır.
        degrees=15.0,
        shear=2.0,
        perspective=0.0003,
        fliplr=0.5,
        flipud=0.0,
    )
    return model


# Eğitilen en iyi ağırlığı mobilde kullanılan çözünürlükte ayrıca doğrular.
def validate_for_mobile(model: YOLO, args: argparse.Namespace) -> None:
    if args.skip_mobile_val:
        return

    trainer = model.trainer
    best_weights = Path(trainer.best) if trainer is not None else None

    # Eğitim sonucu best.pt üretilemediyse yanlış dosyayla validation başlatmayız.
    if best_weights is None or not best_weights.is_file():
        print("[mobil validation] best.pt bulunamadı; mobil kontrol atlandı.")
        return

    print()
    print(f"[mobil validation] {args.mobile_imgsz}x{args.mobile_imgsz} çözünürlükte ölçülüyor...")
    best_model = YOLO(str(best_weights))
    best_model.val(
        data=str(args.data.expanduser().resolve()),
        imgsz=args.mobile_imgsz,
        batch=args.batch,
        device=select_device(args.device),
        workers=args.workers,
        plots=True,
        split="val",
    )


# Script doğrudan çalıştırıldığında eğitim ve mobil validation zincirini yürütür.
def main() -> None:
    args = parse_args()
    trained_model = train_model(args)
    validate_for_mobile(trained_model, args)


if __name__ == "__main__":
    main()
