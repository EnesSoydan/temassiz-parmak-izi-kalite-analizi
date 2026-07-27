from ultralytics import YOLO
model = YOLO(r"final_hands_yolo11n_obb\weights\best.pt")

for i in range(1, 8):
    model.predict(source=rf"YOLO\test_images\img{i}.jpeg", save=True, show=True, conf = 0.55)