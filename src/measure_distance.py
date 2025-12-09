import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
def detect_person_distance2sideedge(frame, margin: int):
    """_summary_

    Args:
        frame: _description_

    Returns:
        _type_: _description_
    """

    # 画像サイズの取得 (高さ, 幅)
    h, w = frame.shape[:2]

    # 2. 推論 (人クラスのみ)
    results = model(frame, classes=[0], verbose=False)
    boxes = results[0].boxes

    # --- 判定ロジック: 端にいるか？ ---
    for box in boxes:
        # 座標を取得 (float -> int変換)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 端判定フラグ
        is_at_edge = False

        # 左端 or 右端に触れているかチェック
        if (x1 < margin) or (x2 > w - margin):
            is_at_edge = True

        # 描画の分岐
        if is_at_edge:
            # 端にいる場合: 赤い枠 + 警告ラベル
            color = (0, 0, 255) # Red
            label = "Too Close to Edge"
        else:
            # 正常: 緑の枠
            color = (0, 255, 0) # Green
            label = "Person"

        # 枠とテキストの描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # マージンエリアを可視化（デバッグ用：グレーの薄い線）
    # 左、右、上、下の境界線を描画
    cv2.line(frame, (margin, 0), (margin, h), (200, 200, 200), 1)
    cv2.line(frame, (w - margin, 0), (w- margin, h), (200, 200, 200), 1)

    return [frame, is_at_edge]

if __name__ == "__main__":
        # 1. モデル読み込み
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")

    # --- 設定値 ---
    margin = 50          # 画面端とみなすピクセル幅

    print("開始します...'q'で終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, is_at_edge = detect_person_distance2sideedge(frame, margin)
        cv2.imshow("Custom Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
