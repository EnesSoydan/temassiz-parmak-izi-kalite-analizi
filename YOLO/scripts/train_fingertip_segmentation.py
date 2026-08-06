"""Temassız distal parmak segmentasyonu için küçük U-Net eğitimi.

Manifest satırı şu alanları taşımalıdır:
{
  "image": "images/capture-001-index.jpg",
  "mask": "masks/capture-001-index.png",
  "person_id": "person-01",
  "split": "train"
}

`split` kişi bazında hazırlanır; aynı kişinin farklı çekimleri farklı kümelere
karıştırılmaz. Maskeler tek kanallı PNG ve foreground değeri 255 olmalıdır.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class FingerprintSegmentationDataset(Dataset):
    def __init__(self, entries: list[dict], image_size: int, training: bool):
        self.entries = entries
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        image = Image.open(entry["image"]).convert("RGB")
        mask = Image.open(entry["mask"]).convert("L")
        image, mask = letterbox_pair(image, mask, self.image_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) >= 127).astype(np.float32)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        if self.training:
            image_tensor, mask_tensor = augment_pair(image_tensor, mask_tensor)
        return image_tensor, mask_tensor


def letterbox_pair(image: Image.Image, mask: Image.Image, target_size: int):
    """Görüntü ve maskeyi oranı bozmadan aynı kare tuvale yerleştirir."""
    if image.size != mask.size:
        raise ValueError(f"Görüntü ve maske boyutu farklı: {image.size} != {mask.size}")
    source_width, source_height = image.size
    scale = min(target_size / source_width, target_size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized_image = image.resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    resized_mask = mask.resize(
        (resized_width, resized_height), Image.Resampling.NEAREST
    )
    image_canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    mask_canvas = Image.new("L", (target_size, target_size), 0)
    left = (target_size - resized_width) // 2
    top = (target_size - resized_height) // 2
    image_canvas.paste(resized_image, (left, top))
    mask_canvas.paste(resized_mask, (left, top))
    return image_canvas, mask_canvas


def augment_pair(image: torch.Tensor, mask: torch.Tensor):
    if random.random() < 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])
    if random.random() < 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])

    if random.random() < 0.75:
        angle = random.uniform(-18.0, 18.0) * np.pi / 180.0
        scale = random.uniform(0.88, 1.12)
        shear = random.uniform(-0.08, 0.08)
        affine = torch.tensor(
            [
                [scale * np.cos(angle), -np.sin(angle) + shear, 0.0],
                [np.sin(angle), scale * np.cos(angle), 0.0],
            ],
            dtype=image.dtype,
        ).unsqueeze(0)
        grid = functional.affine_grid(affine, image.unsqueeze(0).shape, align_corners=False)
        image = functional.grid_sample(
            image.unsqueeze(0), grid, mode="bilinear", padding_mode="reflection", align_corners=False
        ).squeeze(0)
        mask = functional.grid_sample(
            mask.unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=False
        ).squeeze(0)

    if random.random() < 0.75:
        brightness = random.uniform(0.72, 1.28)
        contrast = random.uniform(0.72, 1.25)
        image = ((image - 0.5) * contrast + 0.5) * brightness
    if random.random() < 0.35:
        image = image + torch.randn_like(image) * random.uniform(0.0, 0.035)
    if random.random() < 0.35:
        image = torch.clamp(image, 0.0, 1.0)
        glare = torch.rand(1).item() * 0.35
        image[:, :, image.shape[2] // 3 : image.shape[2] // 2] += glare

    return image.clamp(0.0, 1.0), mask.clamp(0.0, 1.0)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(3, 24)
        self.encoder2 = ConvBlock(24, 48)
        self.encoder3 = ConvBlock(48, 96)
        self.bottleneck = ConvBlock(96, 160)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(160, 96, 2, stride=2)
        self.decoder3 = ConvBlock(192, 96)
        self.up2 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.decoder2 = ConvBlock(96, 48)
        self.up1 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.decoder1 = ConvBlock(48, 24)
        self.output = nn.Conv2d(24, 1, 1)

    def forward(self, inputs):
        first = self.encoder1(inputs)
        second = self.encoder2(self.pool(first))
        third = self.encoder3(self.pool(second))
        bottleneck = self.bottleneck(self.pool(third))
        decoded3 = self.decoder3(torch.cat([self.up3(bottleneck), third], dim=1))
        decoded2 = self.decoder2(torch.cat([self.up2(decoded3), second], dim=1))
        decoded1 = self.decoder1(torch.cat([self.up1(decoded2), first], dim=1))
        return self.output(decoded1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_iou = 0.0
    sample_count = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = functional.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        intersection = (predictions * masks).sum(dim=(1, 2, 3))
        union = ((predictions + masks) > 0).float().sum(dim=(1, 2, 3))
        total_iou += ((intersection + 1.0) / (union + 1.0)).sum().item()
        total_loss += loss.item() * images.shape[0]
        sample_count += images.shape[0]
    return total_loss / max(sample_count, 1), total_iou / max(sample_count, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="480x480 distal parmak U-Net eğitimi")
    parser.add_argument("--manifest", type=Path, required=True, help="PNG maske ve JSON metadata manifesti")
    parser.add_argument("--output", type=Path, default=Path("runs/fingertip_segmentation"))
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_manifest(path: Path):
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError("Manifest boş.")
    if any(entry.get("split") not in {"train", "val"} for entry in entries):
        raise ValueError("Her manifest satırında kişi bazlı train/val split bulunmalı.")
    train_people = {entry.get("person_id") for entry in entries if entry["split"] == "train"}
    val_people = {entry.get("person_id") for entry in entries if entry["split"] == "val"}
    if train_people & val_people:
        raise ValueError("Aynı kişi train ve val kümelerine bölünemez.")
    return entries


def main():
    args = parse_args()
    entries = load_manifest(args.manifest)
    train_entries = [{**entry, "image": str((args.manifest.parent / entry["image"]).resolve()), "mask": str((args.manifest.parent / entry["mask"]).resolve())} for entry in entries if entry["split"] == "train"]
    val_entries = [{**entry, "image": str((args.manifest.parent / entry["image"]).resolve()), "mask": str((args.manifest.parent / entry["mask"]).resolve())} for entry in entries if entry["split"] == "val"]
    device = torch.device(args.device)
    model = SmallUNet().to(device)
    train_loader = DataLoader(FingerprintSegmentationDataset(train_entries, args.imgsz, True), batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(FingerprintSegmentationDataset(val_entries, args.imgsz, False), batch_size=args.batch, shuffle=False, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} train_iou={train_iou:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f}")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "imgsz": args.imgsz, "val_iou": val_iou}, args.output / "best.pt")


if __name__ == "__main__":
    main()
