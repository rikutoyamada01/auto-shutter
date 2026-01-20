import cv2
from detect_circle_gesture import CircleGestureDetector

def detect_person_distance2sideedge_demo():
    """
    CircleGestureDetectorのdetect_edge_proximityメソッドを使用したデモ
    """
    
    # 1. 検出器の初期化
    detector = CircleGestureDetector()
    
    # 2. カメラ開始
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        return

    # --- 設定値 ---
    margin = 50          # 画面端とみなすピクセル幅

    print("開始します...'q'で終了")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 左右反転
            frame = cv2.flip(frame, 1)

            # 3. エッジ検出の実行
            # detect_circle_gesture.py に実装されたメソッドを再利用
            annotated_frame, edges = detector.detect_edge_proximity(frame, margin)
            
            # 結果表示
            if edges:
                # "FAR" is not "TOO CLOSE", so let's change the message prefix based on content
                if "FAR" in edges and len(edges) == 1:
                     prefix = "TOO FAR"
                     color = (255, 0, 0) # Blue-ish for Far?
                else:
                     prefix = "TOO CLOSE"
                     color = (0, 0, 255) # Red for Close
                
                msg = f"{prefix}: {', '.join(edges)}"
                cv2.putText(annotated_frame, msg, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(annotated_frame, "Status: OK", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("MediaPipe Edge Detection Demo", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_person_distance2sideedge_demo()
