"""Mevcut YOLO OBB modelinden segmentasyon için kaba polygonlar üretir.

Bu scriptin ürettiği polygonlar pseudo-label'dır. OBB kutusunun dört köşesi
parmağın gerçek silüeti olmadığı için eğitimden önce elle düzeltilmelidir.
Her OBB tespiti, uygulamadaki canonical ROI'ye benzer bir dikey ROI olarak
kaydedilir ve aynı dörtgen PNG maskeye çevrilir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FINGER_NAMES = {"index", "middle", "ring", "pinky"}


def order_points(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    return ordered.astype(np.float32)


def canonical_quad(points: np.ndarray, height: int, padding: int) -> tuple[np.ndarray, int]:
    ordered = order_points(points)
    lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    long_index = int(np.argmax(lengths))
    source = np.roll(ordered, -long_index, axis=0)
    long_length = float(lengths[long_index])
    short_length = float(min(lengths[(long_index + 1) % 4], lengths[(long_index + 3) % 4]))
    width = max(64, min(height, round(height * short_length / max(long_length, 1.0))))
    target = np.array(
        [
            [padding, height - 1 - padding],
            [padding, padding],
            [width - 1 - padding, padding],
            [width - 1 - padding, height - 1 - padding],
        ],
        dtype=np.float32,
    )
    return source, width


def infer_class_name(result: Any, class_index: int) -> str:
    names = result.names
    if isinstance(names, dict):
        return str(names.get(class_index, class_index)).lower()
    return str(names[class_index]).lower()


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="OBB pseudo polygon ve PNG maske üretir.")
    parser.add_argument("--model", type=Path, required=True, help="Ultralytics .pt OBB modeli")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    images_root = args.images_root.resolve()
    output_root = args.output_root.resolve()
    roi_root = output_root / "images" / "obb-pseudo-roi"
    mask_root = output_root / "masks" / "obb-pseudo-roi"
    annotation_root = output_root / "annotations"
    roi_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path
        for path in sorted(images_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"Görsel bulunamadı: {images_root}")

    model = YOLO(str(args.model.resolve()), task="obb")
    annotations: list[dict[str, Any]] = []
    detection_count = 0
    output_annotations = annotation_root / "obb-pseudo-polygons.json"
    chunk_size = max(1, args.chunk_size)
    is_onnx = args.model.suffix.lower() == ".onnx"
    effective_batch = 1 if is_onnx else max(1, args.batch)
    processed_count = 0
    for chunk_start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[chunk_start : chunk_start + chunk_size]
        predict_kwargs: dict[str, Any] = {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "save": False,
            "verbose": False,
            "stream": not is_onnx,
            "batch": effective_batch,
        }
        if args.device is not None:
            predict_kwargs["device"] = args.device
        if is_onnx:
            result_iterator = (
                (
                    chunk_index,
                    model.predict(source=str(image_path), **predict_kwargs)[0],
                )
                for chunk_index, image_path in enumerate(chunk_paths)
            )
        else:
            result_iterator = enumerate(
                model.predict(
                    source=[str(path) for path in chunk_paths], **predict_kwargs
                )
            )
        for chunk_index, result in result_iterator:
            image_index = chunk_start + chunk_index + 1
            image_path = chunk_paths[chunk_index]
            if result.obb is None or len(result.obb) == 0:
                processed_count += 1
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Görsel okunamadı: {image_path}")
            source_height, source_width = image.shape[:2]
            boxes = result.obb.xyxyxyxy.cpu().numpy()
            classes = result.obb.cls.cpu().numpy().astype(int)
            confidences = result.obb.conf.cpu().numpy()
            source_split = image_path.parent.parent.name
            output_split = {"train": "train", "valid": "val", "test": "test"}.get(
                source_split, "train"
            )
            for detection_index, raw_points in enumerate(boxes):
                class_name = infer_class_name(result, int(classes[detection_index]))
                if class_name not in FINGER_NAMES:
                    continue
                points = np.asarray(raw_points, dtype=np.float32)
                source_quad, output_width = canonical_quad(
                    points, args.imgsz, max(1, round(args.imgsz * 0.04))
                )
                output_height = args.imgsz
                padding = max(1, round(args.imgsz * 0.04))
                target_quad = np.array(
                    [
                        [padding, output_height - 1 - padding],
                        [padding, padding],
                        [output_width - 1 - padding, padding],
                        [output_width - 1 - padding, output_height - 1 - padding],
                    ],
                    dtype=np.float32,
                )
                transform = cv2.getPerspectiveTransform(source_quad, target_quad)
                roi = cv2.warpPerspective(image, transform, (output_width, output_height))
                coarse_source_mask = np.zeros((source_height, source_width), dtype=np.uint8)
                cv2.fillPoly(coarse_source_mask, [np.round(points).astype(np.int32)], 255)
                roi_mask = cv2.warpPerspective(
                    coarse_source_mask,
                    transform,
                    (output_width, output_height),
                    flags=cv2.INTER_NEAREST,
                )

                relative_parent = image_path.parent.parent.name
                stem = f"{image_path.stem}-det{detection_index:02d}-{class_name}"
                image_output = roi_root / relative_parent / f"{stem}.jpg"
                mask_output = mask_root / relative_parent / f"{stem}.png"
                image_output.parent.mkdir(parents=True, exist_ok=True)
                mask_output.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_output), roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
                Image.fromarray(roi_mask, mode="L").save(mask_output, format="PNG", optimize=True)

                points_in_roi = target_quad.tolist()
                annotations.append(
                    {
                        "image": relative_path(image_output, data_root),
                        "mask": relative_path(mask_output, data_root),
                        "polygons": [points_in_roi],
                        "coordinate_space": "pixel",
                        "person_id": f"obb-{source_split}",
                        "split": output_split,
                        "source": "obb-pseudo",
                        "source_image": relative_path(image_path, data_root),
                        "capture_id": image_path.stem,
                        "finger_position": class_name,
                        "confidence": float(confidences[detection_index]),
                        "label_status": "pseudo-obb-box-needs-manual-correction",
                    }
                )
                detection_count += 1
            processed_count += 1
            if image_index % 100 == 0 or image_index == len(image_paths):
                print(f"işlenen={image_index}/{len(image_paths)}, tespit={detection_count}")

        output_annotations.write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"parça tamamlandı: {min(chunk_start + chunk_size, len(image_paths))}/{len(image_paths)}")

    output_annotations.write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Toplam tespit: {detection_count}")
    print(f"Polygon/mask listesi: {output_annotations}")


if __name__ == "__main__":
    main()
