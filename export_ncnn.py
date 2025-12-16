from ultralytics import YOLO

def export_models():
    print("Exporting models to NCNN format...")
    
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
