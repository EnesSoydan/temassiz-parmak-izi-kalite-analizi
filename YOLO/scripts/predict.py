from ultralytics import YOLO
model = YOLO(r"final_hands_yolo11n_obb\weights\best.pt")

model.predict(source=r"YOLO\test_images\img7.jpeg", save=True, show=True, conf = 0.55)