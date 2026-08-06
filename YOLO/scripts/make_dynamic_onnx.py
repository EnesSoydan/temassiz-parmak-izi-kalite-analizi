"""Sabit batch=1 ONNX modelini dinamik batch boyutlu bir kopyaya çevirir."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def make_dynamic(shape) -> None:
    if not shape.dim:
        return
    first = shape.dim[0]
    first.ClearField("dim_value")
    first.dim_param = "batch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input.resolve()))
    for value in model.graph.input:
        make_dynamic(value.type.tensor_type.shape)
    for value in model.graph.output:
        make_dynamic(value.type.tensor_type.shape)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output.resolve()))
    print(f"Dinamik ONNX yazıldı: {args.output}")


if __name__ == "__main__":
    main()
