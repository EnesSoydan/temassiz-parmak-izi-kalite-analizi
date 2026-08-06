"""OBB/pseudo-box etiketlerini SAM2 kutu prompt'u ile gerçek maskeye çevirir.

Bu script mevcut kanonik ROI görsellerindeki kaba polygonun bounding box'ını
SAM2'ye prompt olarak verir. SAM2'nin piksel maskesi contour polygonuna
çevrilir ve YOLO segmentation inceleme paketi oluşturulur.

Çıktı otomatik pseudo-label'dır. SAM2'nin verdiği maskeler eğitim öncesinde
örneklem olarak görsel şekilde doğrulanmalıdır; modelin maske IoU tahmini
kalite sinyali olarak metadata'ya yazılır, gerçek ground truth değildir.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import SAM


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_prompt_box(label_path: Path, width: int, height: int) -> list[float]:
    points: list[tuple[float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        if len(values) < 7 or (len(values) - 1) % 2 != 0:
            raise ValueError(f"Geçersiz YOLO segmentation satırı: {label_path}")
        coords = values[1:]
        points.extend(
            (coords[index] * width, coords[index + 1] * height)
            for index in range(0, len(coords), 2)
        )
    if len(points) < 3:
        raise ValueError(f"Prompt polygonu boş: {label_path}")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [
        max(0.0, min(xs)),
        max(0.0, min(ys)),
        min(float(width - 1), max(xs)),
        min(float(height - 1), max(ys)),
    ]


def mask_polygon(mask: np.ndarray) -> tuple[list[list[float]], float]:
    binary = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("SAM2 maske konturu üretmedi.")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if len(contour) < 3 or area <= 0:
        raise ValueError("SAM2 maske konturu geçersiz.")
    polygon = contour.reshape(-1, 2).astype(float).tolist()
    return polygon, area


def sam_confidence(result: Any) -> float | None:
    boxes = getattr(result, "boxes", None)
    confidence = getattr(boxes, "conf", None)
    if confidence is None or len(confidence) == 0:
        return None
    return float(confidence[0].item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kanonik ROI'lerde SAM2 kutu prompt'u ile segmentation pseudo-label üretir."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="sam2_b.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Mevcut maskeleri ve checkpoint dosyasını kullanarak devam eder.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    metadata_path = dataset / "review-metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"İnceleme metadata dosyası bulunamadı: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = metadata.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("İnceleme metadata entries alanı boş.")
    if args.max_items is not None:
        entries = entries[: args.max_items]

    annotations: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    output_image_root = output / "images"
    output_mask_root = output / "masks"
    output_image_root.mkdir(parents=True, exist_ok=True)
    output_mask_root.mkdir(parents=True, exist_ok=True)

    partial_annotation_path = output / "sam2-polygons.partial.json"
    partial_skipped_path = output / "skipped.partial.json"
    completed_images: set[str] = set()
    if args.resume:
        if partial_annotation_path.is_file():
            annotations = json.loads(partial_annotation_path.read_text(encoding="utf-8"))
        if partial_skipped_path.is_file():
            skipped = json.loads(partial_skipped_path.read_text(encoding="utf-8"))
        completed_images = {
            str(item.get("image")) for item in annotations if item.get("image")
        }

        # İlk çalıştırma checkpoint yazamadan kesildiyse, mevcut PNG maskeleri
        # tekrar okuyup annotation listesini yeniden kur.
        known_images = {str(item.get("image")) for item in annotations}
        for entry in entries:
            image_relative = str(entry["image"])
            if image_relative in known_images:
                continue
            split = str(Path(image_relative).parts[1])
            source_image = (dataset / image_relative).resolve()
            existing_mask = output_mask_root / split / f"{source_image.stem}.png"
            if not existing_mask.is_file():
                continue
            existing = cv2.imread(str(existing_mask), cv2.IMREAD_GRAYSCALE)
            if existing is None:
                continue
            try:
                polygon, area = mask_polygon(existing)
            except ValueError:
                continue
            annotations.append(
                {
                    "image": (output_image_root / split / source_image.name)
                    .relative_to(output)
                    .as_posix(),
                    "mask": existing_mask.relative_to(output).as_posix(),
                    "polygons": [polygon],
                    "coordinate_space": "pixel",
                    "person_id": f"sam2-{split}",
                    "split": split,
                    "source": "sam2-box-prompt",
                    "source_image": entry.get("source_image", image_relative),
                    "capture_id": entry.get("capture_id", source_image.stem),
                    "finger_position": entry.get("finger_position", "unknown"),
                    "sam_confidence": None,
                    "mask_area_ratio": area / max(existing.shape[0] * existing.shape[1], 1),
                    "label_status": "pseudo-sam2-needs-review",
                }
            )
            completed_images.add(image_relative)

    pending_entries = [entry for entry in entries if str(entry["image"]) not in completed_images]
    model = SAM(args.model) if pending_entries else None

    for index, entry in enumerate(entries, start=1):
        if str(entry["image"]) in completed_images:
            continue
        split = str(Path(entry["image"]).parts[1])
        source_image = (dataset / entry["image"]).resolve()
        source_label = (dataset / entry["label"]).resolve()
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(f"Görsel veya label bulunamadı: {source_image}, {source_label}")

        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Görsel okunamadı: {source_image}")
        height, width = image.shape[:2]
        box = load_prompt_box(source_label, width, height)

        predict_kwargs: dict[str, Any] = {
            "bboxes": [box],
            "imgsz": args.imgsz,
            "verbose": False,
        }
        if args.device is not None:
            predict_kwargs["device"] = args.device
        try:
            results = model.predict(source=str(source_image), **predict_kwargs)
            if not results or results[0].masks is None or len(results[0].masks) == 0:
                raise ValueError("SAM2 maske üretmedi.")
            result = results[0]
            mask_data = result.masks.data[0].detach().cpu().numpy()
            polygon, area = mask_polygon(mask_data)
        except ValueError as error:
            skipped.append({"image": str(entry["image"]), "reason": str(error)})
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(np.asarray(polygon)).astype(np.int32)], 255)

        image_output = output_image_root / split / source_image.name
        mask_output = output_mask_root / split / f"{source_image.stem}.png"
        image_output.parent.mkdir(parents=True, exist_ok=True)
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_output)
        Image.fromarray(mask, mode="L").save(mask_output, format="PNG", optimize=True)

        annotations.append(
            {
                "image": image_output.relative_to(output).as_posix(),
                "mask": mask_output.relative_to(output).as_posix(),
                "polygons": [polygon],
                "coordinate_space": "pixel",
                "person_id": f"sam2-{split}",
                "split": split,
                "source": "sam2-box-prompt",
                "source_image": entry.get("source_image", entry["image"]),
                "capture_id": entry.get("capture_id", source_image.stem),
                "finger_position": entry.get("finger_position", "unknown"),
                "sam_confidence": sam_confidence(result),
                "mask_area_ratio": area / max(width * height, 1),
                "label_status": "pseudo-sam2-needs-review",
            }
        )
        if index % 10 == 0 or index == len(entries):
            print(f"işlenen={index}/{len(entries)}, maske={len(annotations)}")
        if index % max(1, args.checkpoint_every) == 0:
            partial_annotation_path.write_text(
                json.dumps(annotations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            partial_skipped_path.write_text(
                json.dumps(skipped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    output.mkdir(parents=True, exist_ok=True)
    annotation_path = output / "sam2-polygons.json"
    annotation_path.write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "skipped.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial_annotation_path.unlink(missing_ok=True)
    partial_skipped_path.unlink(missing_ok=True)
    print(f"SAM2 annotation çıktısı: {annotation_path} ({len(annotations)} örnek)")
    print(f"Atlanan örnek: {len(skipped)}")


if __name__ == "__main__":
    main()
