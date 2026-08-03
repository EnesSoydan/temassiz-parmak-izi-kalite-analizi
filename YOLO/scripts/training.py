"""Zor kamera kosullarina dayanikli YOLO OBB egitim scripti.

Bu script iki farkli augmentation katmani kullanir:

1. Ultralytics'in geometrik augmentation'lari: yatay/dikey el gorunumu,
   donme, perspektif, olcek ve kadraj degisimleri. Bu katman OBB etiketlerini
   kendi icinde beraber donusturur.
2. Egitim batch'i uzerinde calisan fotometrik augmentation'lar: lokal flaş,
   highlight/clipping, golge, kontrast dususu, hafif hareket bulanikligi,
   renk sicakligi degisimi ve sensor benzeri kucuk gurultu. Bu katman sadece
   pikselleri degistirir; OBB koordinatlari degismez.

Ornek:
    python YOLO/scripts/training.py --data "C:/veri/data.yaml"

Daha kucuk bir deneme:
    python YOLO/scripts/training.py --epochs 10 --batch 8 --device 0

Ozel fotometrik augmentation'i kapatip yalnizca Ultralytics augmentation'lari
ile karsilastirma yapmak icin:
    python YOLO/scripts/training.py --disable-photometric-augment

Not: Sentetik augmentation gercek veri eksigini tamamen kapatmaz. Ozellikle
yatay el, flaş ve perspektif iceren bir validation/test parcasi ayri tutulup
gercek cihaz goruntuleriyle kontrol edilmelidir.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as functional
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.obb.train import OBBTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "final_hands_yolo11n_obb" / "weights" / "best.pt"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "fingertip_obb"
DATA_YAML = PROJECT_ROOT / "YOLO" / "datasets" / "merged_hands_fingertip_obb" / "data.yaml"


# Veri setinin fiziksel goruntusunu degistirmeden, telefondaki cekim kosullarini
# batch icinde taklit eden fotometrik augmentation ayarlarini tek yerde tutar.
@dataclass(frozen=True)
class PhotometricAugmentationConfig:
    enabled: bool = True
    global_probability: float = 0.85
    local_flash_probability: float = 0.45
    highlight_probability: float = 0.35
    shadow_probability: float = 0.35
    contrast_probability: float = 0.35
    motion_blur_probability: float = 0.25
    temperature_probability: float = 0.35
    noise_probability: float = 0.20


# Komut satirindan veri seti, baslangic modeli ve augmentation ayarlarini alir.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lokal flaş, clipping, golge, kontrast, hareket bulanikligi, "
            "renk sicakligi ve yatay el augmentation'lariyla YOLO OBB egitimi."
        )
    )
    # Masaustu egitim arayuzu arguman gondermese bile proje icindeki data.yaml kullanilir.
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_YAML,
        help="YOLO data.yaml dosyasi.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL if DEFAULT_MODEL.exists() else "yolo11n-obb.pt"),
        help="Baslangic .pt modeli veya Ultralytics model adi.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Egitim epoch sayisi.")
    parser.add_argument("--batch", type=int, default=12, help="Batch boyutu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Egitim goruntu boyutu.")
    parser.add_argument(
        "--mobile-imgsz",
        type=int,
        default=480,
        help="Mobil uygulamanin 480x480 girdisine yakin validation boyutu.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Veri yukleme worker sayisi.")
    parser.add_argument("--device", default=None, help="Ornek: 0, 0,1 veya cpu.")
    parser.add_argument(
        "--name",
        default="robust_illumination_rotation_obb",
        help="Egitim kosusu adi.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Egitim sonuclarinin yazilacagi klasor.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Veri setini onbellege alir; yeterli RAM varsa kullanilmalidir.",
    )
    parser.add_argument(
        "--skip-mobile-val",
        action="store_true",
        help="Egitim sonundaki mobil validation adimini atlar.",
    )
    parser.add_argument(
        "--disable-photometric-augment",
        action="store_true",
        help="Lokal isik, golge, blur, kontrast ve renk augmentation'larini kapatir.",
    )
    parser.add_argument(
        "--photo-augment-prob",
        type=float,
        default=0.85,
        help="Fotometrik augmentation katmaninin genel olasiligi.",
    )
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=90.0,
        help="Elin yatay/dikey gorunumunu taklit eden maksimum donme acisi.",
    )
    parser.add_argument(
        "--local-flash-prob",
        type=float,
        default=0.45,
        help="Lokal parlak beyaz bolge augmentation olasiligi.",
    )
    parser.add_argument(
        "--highlight-prob",
        type=float,
        default=0.35,
        help="Highlight/clipping augmentation olasiligi.",
    )
    parser.add_argument(
        "--shadow-prob",
        type=float,
        default=0.35,
        help="Golge augmentation olasiligi.",
    )
    parser.add_argument(
        "--contrast-prob",
        type=float,
        default=0.35,
        help="Kontrast dususu augmentation olasiligi.",
    )
    parser.add_argument(
        "--motion-blur-prob",
        type=float,
        default=0.25,
        help="Hafif yonlu hareket bulanikligi augmentation olasiligi.",
    )
    parser.add_argument(
        "--temperature-prob",
        type=float,
        default=0.35,
        help="Renk sicakligi degisimi augmentation olasiligi.",
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=0.20,
        help="Hafif sensor gurultusu augmentation olasiligi.",
    )
    return parser.parse_args()


# Olasilik parametrelerini guvenli araliga alir; hatali komut satiri degerleri
# augmentation katmaninin egitimi tamamen bozmasini engeller.
def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


# data.yaml icerigini okuyup egitim baslamadan once temel dosya hatalarini yakalar.
def load_dataset_config(data_yaml: Path) -> dict:
    resolved_yaml = data_yaml.expanduser().resolve()

    # Yanlis yol verilirse Ultralytics'in uzun hata mesajindan once sebebi gosteririz.
    if not resolved_yaml.is_file():
        raise FileNotFoundError(f"data.yaml bulunamadi: {resolved_yaml}")

    config = yaml.safe_load(resolved_yaml.read_text(encoding="utf-8"))

    # Egitim icin sinif isimleri ve train yolu mutlaka bulunmalidir.
    if not isinstance(config, dict) or "train" not in config or "names" not in config:
        raise ValueError("data.yaml icinde en az train ve names alanlari bulunmali.")

    return config


# data.yaml icindeki path alanini yaml konumuna gore mutlak veri seti yoluna cevirir.
def resolve_dataset_root(data_yaml: Path, config: dict) -> Path:
    configured_root = config.get("path")

    # path verilmemisse train/valid klasorlerinin data.yaml yaninda oldugunu kabul ederiz.
    if configured_root in (None, ""):
        return data_yaml.expanduser().resolve().parent

    root = Path(str(configured_root)).expanduser()

    # Goreli path degerleri data.yaml dosyasinin bulundugu klasore gore cozulur.
    if not root.is_absolute():
        root = data_yaml.expanduser().resolve().parent / root

    return root.resolve()


# Goruntu split yolundan ayni yapida bulunan labels klasorunu bulur.
def resolve_labels_directory(dataset_root: Path, split_value: str) -> Path | None:
    split_path = Path(split_value)

    # URL veya Ultralytics'in uzak veri kaynaklari yerel etiket analizine uygun degildir.
    if "://" in split_value:
        return None

    image_path = split_path if split_path.is_absolute() else dataset_root / split_path
    parts = list(image_path.parts)

    # Standart split/images yapisini split/labels bicimine donustururuz.
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts)

    return image_path.parent / "labels"


# Normalize OBB kose noktalarindan goruntu alanina oranli poligon alanini hesaplar.
def calculate_obb_area(values: list[float]) -> float:
    points = [(values[index], values[index + 1]) for index in range(0, 8, 2)]
    doubled_area = 0.0

    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        doubled_area += point[0] * next_point[1] - next_point[0] * point[1]

    return abs(doubled_area) / 2


# Egitim etiketlerindeki kucuk parmak ucu oranini raporlayarak uzaklik veri
# acigini gorunur yapar.
def report_object_scale_distribution(data_yaml: Path, config: dict) -> None:
    dataset_root = resolve_dataset_root(data_yaml, config)
    train_value = config.get("train")

    # Birden fazla train kaynagi varsa ilk yerel kaynagi analiz ederiz.
    if isinstance(train_value, list):
        train_value = next((value for value in train_value if isinstance(value, str)), None)

    if not isinstance(train_value, str):
        print("[olcek analizi] Train yolu cozulmedi; etiket alan analizi atlandi.")
        return

    labels_dir = resolve_labels_directory(dataset_root, train_value)

    # data.yaml baska bir makinenin mutlak yolunu tasiyorsa uyari veririz.
    if labels_dir is None or not labels_dir.is_dir():
        print(f"[olcek analizi] Labels klasoru bulunamadi: {labels_dir}")
        print("[olcek analizi] data.yaml icindeki path/train yollarini kontrol et.")
        return

    areas: list[float] = []
    invalid_lines = 0

    for label_path in labels_dir.rglob("*.txt"):
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()

            # OBB etiketi class_id ve sekiz normalize koordinattan olusmalidir.
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

    # Dolu OBB etiketi yoksa egitimden once acik bir uyari uretiriz.
    if not areas:
        print("[olcek analizi] Gecerli OBB etiketi bulunamadi.")
        return

    very_small_count = sum(area < 0.005 for area in areas)
    small_count = sum(area < 0.015 for area in areas)
    very_small_ratio = very_small_count / len(areas)
    small_ratio = small_count / len(areas)

    print("[olcek analizi]")
    print(f"  toplam OBB etiketi: {len(areas)}")
    print(f"  medyan kutu alani: %{median(areas) * 100:.3f}")
    print(f"  cok kucuk kutu (<%0.5): {very_small_count} (%{very_small_ratio * 100:.1f})")
    print(f"  kucuk kutu (<%1.5): {small_count} (%{small_ratio * 100:.1f})")
    print(f"  bozuk/OBB olmayan satir: {invalid_lines}")

    # Uzak cekim basarisi icin augmentation'in yaninda gercek kucuk el ornekleri
    # de bulunmalidir.
    if small_ratio < 0.15:
        print(
            "  UYARI: Kucuk parmak ucu orani dusuk. "
            "Veri setine gercekten uzaktan cekilmis ve elle dogrulanmis ornekler ekle."
        )


# CUDA varsa ilk GPU'yu, yoksa CPU'yu secer; --device bunu degistirebilir.
def select_device(requested_device: str | None) -> str:
    if requested_device:
        return requested_device

    return "0" if torch.cuda.is_available() else "cpu"


# Batch icindeki her goruntu icin ayni boyutta normalize koordinat izgara uretir.
def make_normalized_grid(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    y_values = torch.linspace(-1.0, 1.0, height, device=device)
    x_values = torch.linspace(-1.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(y_values, x_values, indexing="ij")
    return grid_x, grid_y


# Her goruntu icin el uzerinde lokal, yumusak eliptik bir etki maskesi uretir.
def create_ellipse_mask(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    batch_size: int,
    center_x_range: tuple[float, float],
    center_y_range: tuple[float, float],
    radius_x_range: tuple[float, float],
    radius_y_range: tuple[float, float],
) -> torch.Tensor:
    device = grid_x.device
    center_x = torch.empty(batch_size, 1, 1, device=device).uniform_(*center_x_range)
    center_y = torch.empty(batch_size, 1, 1, device=device).uniform_(*center_y_range)
    radius_x = torch.empty(batch_size, 1, 1, device=device).uniform_(*radius_x_range)
    radius_y = torch.empty(batch_size, 1, 1, device=device).uniform_(*radius_y_range)

    distance = ((grid_x.unsqueeze(0) - center_x) / radius_x).square()
    distance = distance + ((grid_y.unsqueeze(0) - center_y) / radius_y).square()
    return torch.exp(-0.5 * distance).unsqueeze(1)


# Yalnizca secilen batch orneklerine efekt uygular; diger ornekleri aynen korur.
def select_samples(
    original: torch.Tensor,
    transformed: torch.Tensor,
    enabled: torch.Tensor,
) -> torch.Tensor:
    return torch.where(enabled.view(-1, 1, 1, 1), transformed, original)


# Lokal flaş veya parlak beyaz bolgeyi, parmak yuzeyinin bir kismini beyaza
# yaklastirarak simule eder.
def apply_local_flash(
    images: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    enabled: torch.Tensor,
) -> torch.Tensor:
    batch_size = images.shape[0]
    mask = create_ellipse_mask(
        grid_x,
        grid_y,
        batch_size,
        center_x_range=(-0.45, 0.45),
        center_y_range=(-0.35, 0.40),
        radius_x_range=(0.16, 0.42),
        radius_y_range=(0.14, 0.38),
    )
    strength = torch.empty(batch_size, 1, 1, 1, device=images.device).uniform_(0.28, 0.68)
    transformed = images + (1.0 - images) * mask * strength
    return select_samples(images, transformed, enabled)


# Highlight clipping'i, hotspot icindeki en parlak pikselleri gercekten 1.0'a
# kirparak simule eder; model bu durumda parmak sinyalinin kaybolabilecegini ogrenir.
def apply_highlight_clipping(
    images: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    enabled: torch.Tensor,
) -> torch.Tensor:
    batch_size = images.shape[0]
    mask = create_ellipse_mask(
        grid_x,
        grid_y,
        batch_size,
        center_x_range=(-0.40, 0.40),
        center_y_range=(-0.30, 0.35),
        radius_x_range=(0.06, 0.20),
        radius_y_range=(0.06, 0.20),
    )
    strength = torch.empty(batch_size, 1, 1, 1, device=images.device).uniform_(0.50, 0.92)
    transformed = images + (1.0 - images) * mask * strength
    threshold = torch.empty(batch_size, 1, 1, 1, device=images.device).uniform_(0.65, 0.86)
    transformed = torch.where(mask > threshold, torch.ones_like(transformed), transformed)
    return select_samples(images, transformed, enabled)


# Genis ve yumusak bir kararma maskesiyle elin bir bolgesindeki golgeyi simule eder.
def apply_shadow(
    images: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    enabled: torch.Tensor,
) -> torch.Tensor:
    batch_size = images.shape[0]
    mask = create_ellipse_mask(
        grid_x,
        grid_y,
        batch_size,
        center_x_range=(-0.65, 0.65),
        center_y_range=(-0.65, 0.65),
        radius_x_range=(0.35, 0.95),
        radius_y_range=(0.30, 0.90),
    )
    strength = torch.empty(batch_size, 1, 1, 1, device=images.device).uniform_(0.18, 0.48)
    transformed = images * (1.0 - strength * mask)
    return select_samples(images, transformed, enabled)


# Kontrast dususunu lokal ortalama etrafinda daraltarak simule eder; parlakligi
# tamamen sifirlamaz ki efekt fotografik olarak asiri yapaylasmasin.
def apply_contrast_drop(images: torch.Tensor, enabled: torch.Tensor) -> torch.Tensor:
    batch_size = images.shape[0]
    factor = torch.empty(batch_size, 1, 1, 1, device=images.device).uniform_(0.52, 0.88)
    offset = torch.empty(batch_size, 1, 1, 1, device=images.device).uniform_(-0.035, 0.035)
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    transformed = (images - mean) * factor + mean + offset
    return select_samples(images, transformed, enabled)


# Sicak ve soguk beyaz dengesi degisimlerini RGB kanal katsayilariyla simule eder.
def apply_color_temperature(images: torch.Tensor, enabled: torch.Tensor) -> torch.Tensor:
    batch_size = images.shape[0]
    warm = torch.rand(batch_size, 1, 1, 1, device=images.device) > 0.5
    strength = torch.empty(batch_size, 1, 1, 1, device=images.device).uniform_(0.06, 0.16)
    ones = torch.ones_like(strength)
    warm_scale = torch.cat((ones + strength, ones, ones - strength), dim=1)
    cool_scale = torch.cat((ones - strength, ones, ones + strength), dim=1)
    scale = torch.where(warm, warm_scale, cool_scale)
    transformed = images * scale
    return select_samples(images, transformed, enabled)


# Dikey, yatay veya capraz yonlu kisa kernel kullanarak hafif hareket blur'u uygular.
def apply_directional_blur_to_sample(image: torch.Tensor, kernel_size: int, direction: int) -> torch.Tensor:
    channels = image.shape[0]
    kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=image.device, dtype=image.dtype)
    center = kernel_size // 2

    if direction == 0:
        kernel[0, 0, center, :] = 1.0
    elif direction == 1:
        kernel[0, 0, :, center] = 1.0
    elif direction == 2:
        indices = torch.arange(kernel_size, device=image.device)
        kernel[0, 0, indices, indices] = 1.0
    else:
        indices = torch.arange(kernel_size, device=image.device)
        kernel[0, 0, indices, kernel_size - 1 - indices] = 1.0

    kernel /= kernel.sum()
    grouped_kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    return functional.conv2d(
        image.unsqueeze(0),
        grouped_kernel,
        padding=kernel_size // 2,
        groups=channels,
    ).squeeze(0)


# Hareket blur'unu batch'te secilen goruntulere uygular; batch boyutu kucuk oldugu
# icin bu acik dongu, pahali bir harici goruntu kutuphanesinden daha okunabilirdir.
def apply_motion_blur(images: torch.Tensor, enabled: torch.Tensor) -> torch.Tensor:
    transformed = images.clone()
    selected_indices = torch.nonzero(enabled, as_tuple=False).flatten().tolist()

    for index in selected_indices:
        kernel_size = 3 if torch.rand(()) < 0.65 else 5
        direction = int(torch.randint(0, 4, ()).item())
        transformed[index] = apply_directional_blur_to_sample(
            transformed[index], int(kernel_size), direction
        )

    return select_samples(images, transformed, enabled)


# Kamera sensorundaki hafif gurultuyu ekler; clipping veya parlama gibi buyuk
# kusurlarin yerini tutmaz, yalnizca temiz dataset'e kucuk cihaz varyasyonu ekler.
def apply_sensor_noise(images: torch.Tensor, enabled: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(images) * 0.012
    transformed = images + noise
    return select_samples(images, transformed, enabled)


# Tum fotometrik augmentation'lari sadece egitim batch'ine uygular. Bu fonksiyon
# validation verisini degistirmez ve son adimda piksel araligini 0..1'e sabitler.
def augment_photometric_batch(
    images: torch.Tensor,
    config: PhotometricAugmentationConfig,
) -> torch.Tensor:
    if not config.enabled or images.ndim != 4 or images.shape[1] < 3:
        return images

    with torch.no_grad():
        images = images.float()
        batch_size, _, height, width = images.shape
        grid_x, grid_y = make_normalized_grid(height, width, images.device)
        general_probability = clamp_probability(config.global_probability)

        # Her efekt bagimsiz secilir; boylece model tek bir sabit bozulma kalibina
        # ezberlenmek yerine gercek telefon kosullarinin kombinasyonlarini gorur.
        local_flash_enabled = torch.rand(batch_size, device=images.device) < general_probability * clamp_probability(
            config.local_flash_probability
        )
        highlight_enabled = torch.rand(batch_size, device=images.device) < general_probability * clamp_probability(
            config.highlight_probability
        )
        shadow_enabled = torch.rand(batch_size, device=images.device) < general_probability * clamp_probability(
            config.shadow_probability
        )
        contrast_enabled = torch.rand(batch_size, device=images.device) < general_probability * clamp_probability(
            config.contrast_probability
        )
        blur_enabled = torch.rand(batch_size, device=images.device) < general_probability * clamp_probability(
            config.motion_blur_probability
        )
        temperature_enabled = torch.rand(batch_size, device=images.device) < general_probability * clamp_probability(
            config.temperature_probability
        )
        noise_enabled = torch.rand(batch_size, device=images.device) < general_probability * clamp_probability(
            config.noise_probability
        )

        images = apply_local_flash(images, grid_x, grid_y, local_flash_enabled)
        images = apply_highlight_clipping(images, grid_x, grid_y, highlight_enabled)
        images = apply_shadow(images, grid_x, grid_y, shadow_enabled)
        images = apply_contrast_drop(images, contrast_enabled)
        images = apply_color_temperature(images, temperature_enabled)
        images = apply_motion_blur(images, blur_enabled)
        images = apply_sensor_noise(images, noise_enabled)

        return images.clamp_(0.0, 1.0)


# Ultralytics'in OBB trainer'ini override ederek fotometrik augmentation'i
# geometrik/etiket donusumlerinden sonra ve model kaybindan hemen once calistirir.
class RobustOBBTrainer(OBBTrainer):
    photometric_config = PhotometricAugmentationConfig()

    # Super sinif once goruntuyu cihaza tasir ve 0..255 araligindan 0..1'e cevirir.
    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        batch["img"] = augment_photometric_batch(batch["img"], self.photometric_config)
        return batch


# Komut satiri ayarlarini custom trainer sinifina aktarir; Ultralytics tarafina
# bilinmeyen config anahtarlari gondermeden augmentation'i etkinlestirir.
def configure_trainer(args: argparse.Namespace) -> type[RobustOBBTrainer]:
    RobustOBBTrainer.photometric_config = PhotometricAugmentationConfig(
        enabled=not args.disable_photometric_augment,
        global_probability=clamp_probability(args.photo_augment_prob),
        local_flash_probability=clamp_probability(args.local_flash_prob),
        highlight_probability=clamp_probability(args.highlight_prob),
        shadow_probability=clamp_probability(args.shadow_prob),
        contrast_probability=clamp_probability(args.contrast_prob),
        motion_blur_probability=clamp_probability(args.motion_blur_prob),
        temperature_probability=clamp_probability(args.temperature_prob),
        noise_probability=clamp_probability(args.noise_prob),
    )
    return RobustOBBTrainer


# Baslangic modeliyle robust OBB egitimini baslatir.
def train_model(args: argparse.Namespace) -> YOLO:
    data_yaml = args.data.expanduser().resolve()
    config = load_dataset_config(data_yaml)
    report_object_scale_distribution(data_yaml, config)
    trainer_class = configure_trainer(args)
    augmentation_config = trainer_class.photometric_config

    print()
    print(f"Baslangic modeli: {args.model}")
    print(f"Veri seti: {data_yaml}")
    print(f"Cihaz: {select_device(args.device)}")
    print(f"Donme acisi: +/-{args.rotation_degrees:.1f} derece")
    print(
        "Fotometrik augmentation: "
        f"{'aktif' if augmentation_config.enabled else 'kapali'} "
        f"(genel olasilik={augmentation_config.global_probability:.2f})"
    )

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
        # Ultralytics'in standart renk ve isik augmentation'lari temel varyasyonu saglar.
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.60,
        # Uzak/yakin el olcegini ve kadraj icindeki konum degisimini taklit eder.
        scale=0.70,
        translate=0.20,
        multi_scale=0.25,
        # Farkli el yonlerini ve telefon perspektifini taklit eder; OBB etiketleri
        # Ultralytics tarafindan bu donusumlerle birlikte guncellenir.
        degrees=args.rotation_degrees,
        shear=8.0,
        perspective=0.0008,
        fliplr=0.5,
        flipud=0.0,
        # Dusuk oranli mosaic kucuk nesne gorunumunu artirir, son epochlarda kapanir.
        mosaic=0.35,
        mixup=0.0,
        # Fotometrik augmentation'in uygulanacagi custom OBB trainer.
        trainer=trainer_class,
    )
    return model


# Egitilen en iyi agirligi mobilde kullanilan cozumurlukte ayrica dogrular.
def validate_for_mobile(model: YOLO, args: argparse.Namespace) -> None:
    if args.skip_mobile_val:
        return

    trainer = model.trainer
    best_weights = Path(trainer.best) if trainer is not None else None

    # Egitim sonucu best.pt uretilemediyse validation baslatmayiz.
    if best_weights is None or not best_weights.is_file():
        print("[mobil validation] best.pt bulunamadi; mobil kontrol atlandi.")
        return

    print()
    print(f"[mobil validation] {args.mobile_imgsz}x{args.mobile_imgsz} cozulurlukte olculuyor...")
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


# Script dogrudan calistirildiginda egitim ve mobil validation zincirini yurutur.
def main() -> None:
    args = parse_args()
    trained_model = train_model(args)
    validate_for_mobile(trained_model, args)


if __name__ == "__main__":
    main()
