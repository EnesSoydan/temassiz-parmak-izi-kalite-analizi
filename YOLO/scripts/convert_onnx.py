from ultralytics import YOLO

model = YOLO(r"runs\fingertip_obb\robust_illumination_rotation_obb\weights\best.pt")

model.export(
    format="onnx",
    imgsz=480,
    simplify=True,
    opset=12,
    dynamic = False,
)
