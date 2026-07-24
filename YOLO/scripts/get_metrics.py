from ultralytics import YOLO

def get_metrics(model_path, data_yaml):
    """
    Function to load a YOLO model, run validation, and print overall and per-class metrics.
    
    Parameters:
    - model_path: str, path to the YOLO model file.
    - data_yaml: str, path to the data configuration YAML file.
    
    Returns:
    - None
    """
    #Load model
    model = YOLO("model_path")

    #Run validation on the model
    metrics = model.val(data="data.yaml", split = "val")

    #Access overall (mean) metrics for precision and recall
    mean_precision = metrics.results_dict['metrics/precision(B)']
    mean_recall = metrics.results_dict['metrics/recall(B)']
    f1_score = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0.0

    print(f"Overall Precision: {mean_precision:.4f}")

    print(f"Overall Recall: {mean_recall:.4f}")

    print(f"F1 Score: {f1_score:.4f}")

    #Access per-class metrics
    for i, class_name in enumerate(metrics.names.values()):
        #class result returns [precision, recall, mAP50, mAP50-95]
        p,r,_,_ = metrics.class_result(i)
        print(f"Class: [{class_name}] -> Precision: {p:.4f}, Recall: {r:.4f}")



def visualize_metrics():
    """
    Function to visualize training and validation metrics from the results CSV file.
    
    Returns:
    - None
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    # 1. Load the automatically saved CSV results file
    results_df = pd.read_csv('runs/detect/train/results.csv')

    # Clean column headers (strip extra spaces)
    results_df.columns = results_df.columns.str.strip()

    # 2. Plot Training & Validation Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['epoch'], results_df['train/box_loss'], label='Train Box Loss', color='blue')
    plt.plot(results_df['epoch'], results_df['val/box_loss'], label='Val Box Loss', linestyle='--', color='orange')
    plt.title('YOLO Box Loss Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Plot Mean Average Precision (mAP) Curve
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='mAP @ 0.50', color='green')
    plt.plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], label='mAP @ 0.50:0.95', color='red')
    plt.title('YOLO Model mAP Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/yolo_model.pt"  # Replace with your model path
    data_yaml = "path/to/your/data.yaml"        # Replace with your data YAML path

    get_metrics(model_path, data_yaml)
    visualize_metrics()