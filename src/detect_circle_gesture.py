import cv2
import math
from ultralytics import YOLO

# モデルのロード
pose_model = YOLO("yolo11n-pose.pt")

def detect_circle_gesture(frame):
    """
    条件:
    1. 両手首が両肘より上
    2. 両肘が両肩より上
    3. 両手首が近づいている
    """
    
    # 1. 推論
    results = pose_model(frame, verbose=False)
    draw_frame = frame.copy()
    detected_flag = 0

    if results[0].keypoints is not None and results[0].keypoints.data.shape[1] > 0:
        for keypoints in results[0].keypoints.data:
            kpts = keypoints.cpu().numpy()

            # --- 座標の取得 ---
            # インデックス: 5,6=肩, 7,8=肘, 9,10=手首
            l_shoulder = kpts[5]
            r_shoulder = kpts[6]
            l_elbow = kpts[7]
            r_elbow = kpts[8]
            l_wrist = kpts[9]
            r_wrist = kpts[10]

            # --- 信頼度チェック (0.5未満ならスキップ) ---
            if (l_shoulder[2] < 0.5 or r_shoulder[2] < 0.5 or
                l_elbow[2] < 0.5 or r_elbow[2] < 0.5 or
                l_wrist[2] < 0.5 or r_wrist[2] < 0.5):
                continue

            # --- 描画 (関節とボーン) ---
            # 視覚化のため、座標を整数に変換
            joints_coords = {}
            for i, name in [(5, 'ls'), (6, 'rs'), (7, 'le'), (8, 're'), (9, 'lw'), (10, 'rw')]:
                x, y = int(kpts[i][0]), int(kpts[i][1])
                joints_coords[name] = (x, y)
                # 関節を丸で描画
                cv2.circle(draw_frame, (x, y), 6, (0, 255, 255), -1)

            # 腕の線を描画
            if 'ls' in joints_coords and 'le' in joints_coords:
                cv2.line(draw_frame, joints_coords['ls'], joints_coords['le'], (0, 255, 0), 2)
            if 'le' in joints_coords and 'lw' in joints_coords:
                cv2.line(draw_frame, joints_coords['le'], joints_coords['lw'], (0, 255, 0), 2)
            if 'rs' in joints_coords and 're' in joints_coords:
                cv2.line(draw_frame, joints_coords['rs'], joints_coords['re'], (0, 255, 0), 2)
            if 're' in joints_coords and 'rw' in joints_coords:
                cv2.line(draw_frame, joints_coords['re'], joints_coords['rw'], (0, 255, 0), 2)


            # --- 判定ロジック ---
            
            # Y座標は画面上が0なので、「上にある」＝「値が小さい」
            
            # 条件1: 手首が肘より上
            cond_wrists_above_elbows = (l_wrist[1] < l_elbow[1]) and (r_wrist[1] < r_elbow[1])
            
            # 条件2: 肘が肩より上
            cond_elbows_above_shoulders = (l_elbow[1] < l_shoulder[1]) and (r_elbow[1] < r_shoulder[1])
            
            # 条件3: 手首同士が近づいているか
            # 基準として肩幅を使用
            wrist_dist = math.hypot(l_wrist[0] - r_wrist[0], l_wrist[1] - r_wrist[1])
            shoulder_width = math.hypot(l_shoulder[0] - r_shoulder[0], l_shoulder[1] - r_shoulder[1])
            
            # 「近づいている」の定義: 肩幅と同じか、それより狭い距離にあればOKとする
            # (少し広くてもOKにしたい場合は 1.0 や 1.2 に調整してください)
            cond_wrists_close = wrist_dist < (shoulder_width * 1.2)

            # すべての条件を満たすか
            if cond_wrists_above_elbows and cond_elbows_above_shoulders and cond_wrists_close:
                detected_flag = 1
                
                # 検出時のフィードバック描画
                cv2.putText(draw_frame, "MARU (CIRCLE) DETECTED!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # 検出された人の手首同士を結ぶ線を描画
                cv2.line(draw_frame, joints_coords['lw'], joints_coords['rw'], (0, 0, 255), 4)

    return [draw_frame, detected_flag]

# --- テスト用メイン関数 ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("'q'キーで終了")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        result_frame, status = detect_circle_gesture(frame)
        
        if status == 1:
            print("OKジェスチャー検知！")
            
        cv2.imshow("Pose Detection", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
