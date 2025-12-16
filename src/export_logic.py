import sys
import os

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' module not found.")
    print("Please ensure you are running this script within the virtual environment.")
    sys.exit(1)

def export_models():
    print("Exporting models to NCNN format...")
    
    # Check if model files exist
    if not os.path.exists("yolo11n-pose.pt"):
        print("Error: yolo11n-pose.pt not found.")
        return
    if not os.path.exists("yolo11n.pt"):
        print("Error: yolo11n.pt not found.")
        return

    # 1. Pose Model
    print("Exporting yolo11n-pose.pt...")
    model_pose = YOLO("yolo11n-pose.pt")
    model_pose.export(format="ncnn")
    
    # 2. Detection Model
    print("Exporting yolo11n.pt...")
    model_det = YOLO("yolo11n.pt")
    model_det.export(format="ncnn")
    
    print("Export complete. '_ncnn_model' folders created.")

if __name__ == "__main__":
    export_models()
