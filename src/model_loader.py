import os
from ultralytics import YOLO

def load_model(model_basename: str, task: str = None):
    """
    モデルをロードするヘルパー関数。
    NCNNフォーマットのモデルディレクトリ（{model_basename}_ncnn_model）が存在すればそれを読み込み、
    なければ通常のPtモデル（{model_basename}.pt）を読み込む。

    Args:
        model_basename (str): 拡張子なしのモデル名 (例: "yolo11n-pose")
        task (str, optional): タスク名 ("pose", "detect"など)。NCNNロード時に推奨される。

    Returns:
        YOLO: ロードされたモデルインスタンス
    """
    
    # NCNNモデルのパス (export_ncnn.pyで生成されるフォルダ名)
    ncnn_path = f"{model_basename}_ncnn_model"
    pt_path = f"{model_basename}.pt"

    # カレントディレクトリからの相対パス、もしくは絶対パスの考慮が必要だが
    # ここでは実行ディレクトリ直下を想定
    
    if os.path.exists(ncnn_path):
        print(f"[ModelLoader] NCNN model found: {ncnn_path}")
        # NCNNモデルのロード
        # task引数はNCNNの場合に警告抑制のために指定推奨
        return YOLO(ncnn_path, task=task)
    else:
        print(f"[ModelLoader] NCNN model not found. Falling back to PT: {pt_path}")
        return YOLO(pt_path)
