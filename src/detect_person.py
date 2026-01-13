import cv2
import mediapipe as mp

def detect_person(source=0):
    """
    MediaPipe Poseを使用して動画から人を検出する
    source: 動画ファイルのパス または カメラID (例: 0)
    """
    
    # 1. モデルの初期化 (MediaPipe Pose)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0, # 0=Lite, 1=Full, 2=Heavy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 2. 動画ソースを開く
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("エラー: 動画またはカメラを開けませんでした。")
        return

    print("MediaPipe検出を開始します... 'q'キーで終了します。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 3. 前処理 (BGR -> RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 4. 推論を実行
            results = pose.process(frame_rgb)

            # 5. 結果の描画
            annotated_frame = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # 検出ログ（オプション）
                cv2.putText(annotated_frame, "Person Detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 6. 結果を表示
            cv2.imshow("MediaPipe Person Detection", annotated_frame)

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    # Webカメラの場合は 0、動画ファイルの場合はファイルパスを指定
    detect_person(0)
