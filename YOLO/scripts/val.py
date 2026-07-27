from ultralytics import YOLO    
model = YOLO(r"runs\fingertip_obb\brightness_distance_obb\weights\best.pt") # load a pretrained model (recommended for training)

def val(model):
    model.val(
        data = r"C:\Users\eness\Downloads\merged_hands_fingertip_obb\data.yaml",
        imgsz = 640,
        iou = 0.65,
        conf = 0.5,
        task = "val",
        device = 0,
        workers = 8,
        save = True,
        save_txt = True,
        save_conf = True,
        save_json = True,
        plots = True,
        verbose = True,
        save_dir = r"runs\fingertip_obb\brightness_distance_obb\val_results")

if __name__ == "__main__":
    val(model)