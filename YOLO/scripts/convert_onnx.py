from ultralytics import YOLO

model = YOLO("final_hands_yolo11n_obb/weights/best.pt")

model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    opset=12,
)