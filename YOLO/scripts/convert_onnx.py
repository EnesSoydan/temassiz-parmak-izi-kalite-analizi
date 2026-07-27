from ultralytics import YOLO

model = YOLO(r"runs\fingertip_obb\brightness_distance_obb\weights\best.pt")

model.export(
    format="onnx",
    imgsz=480,
    simplify=True,
    opset=12,
)