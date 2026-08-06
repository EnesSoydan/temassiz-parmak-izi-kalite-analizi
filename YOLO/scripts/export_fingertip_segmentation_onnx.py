"""SmallUNet checkpoint'ini mobil ONNX modeline çevirir."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_fingertip_segmentation import SmallUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=480)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SmallUNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["images"],
        output_names=["mask_logits"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"ONNX yazıldı: {args.output}")


if __name__ == "__main__":
    main()
