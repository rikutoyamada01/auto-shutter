import os
from ultralytics import YOLO

def load_model(model_basename: str, task: str = None):
    """
    モデルをロードするヘルパー関数。
    通常のPtモデル（{model_basename}.pt）を読み込む。

    Args:
        model_basename (str): 拡張子なしのモデル名 (例: "yolo11n-pose")
        task (str, optional): タスク名 ("pose", "detect"など)。

    Returns:
        YOLO: ロードされたモデルインスタンス
    """
    
    pt_path = f"{model_basename}.pt"
    print(f"[ModelLoader] Loading PT model: {pt_path}")
    return YOLO(pt_path)
