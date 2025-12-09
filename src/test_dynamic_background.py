# -*- coding: utf-8 -*-
import cv2
import sys
import time
import numpy as np
from background_subtractor import FixedBackgroundSubtractor, AdaptiveBackgroundSubtractor

# グローバル変数として初期背景画像を保持
initial_far_background = None
# トラックバーの値 (0-100) をスケールファクターに変換するための係数
# 例: 0 -> 1.0 (等倍), 100 -> 2.0 (2倍)
SCALE_FACTOR_MAX = 2.0

def on_trackbar_change(val):
    # トラックバーの値が変更されたときに呼ばれるが、ここでは何もしない
    # 値は直接 cv2.getTrackbarPos で取得する
    pass

def main():
    """
    カメラからの映像をリアルタイムで表示し、
    トラックバーでシミュレートしたカメラ位置に基づいて背景画像を動的に生成し、
    背景差分を検出する概念実証プログラム。
    """
    print("--- 動的背景生成の概念実証プログラム開始 ---")
    
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"エラー: カメラ(インデックス: {camera_index})を開けませんでした。")
        sys.exit(1)

    print(f"カメラ(インデックス: {camera_index})を正常に開きました。")
    
    # --- カメラの自動露出を無効化 ---
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    cap.set(cv2.CAP_PROP_EXPOSURE, 80) 
    print("カメラの自動露出を無効化しました。")
    # --------------------------------

    window_name = "Camera Feed (PoC)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # トラックバーの作成
    trackbar_name = "Camera Position (0=Far, 100=Close)"
    cv2.createTrackbar(trackbar_name, window_name, 0, 100, on_trackbar_change)
    
    print(f"映像ウィンドウ '{window_name}' を表示します。")
    print("ウィンドウを選択した状態で 'q' または 'ESC' キーを押すと終了します。")
    print("トラックバーを動かして、背景画像の変形と差分結果を確認してください。")

    # --- 最初の「遠景」背景画像をキャプチャ ---
    print("最初の「遠景」背景画像をキャプチャ中...")
    for _ in range(30): # ウォームアップ
        cap.read()
    
    ret, bg_frame_initial = cap.read()
    if not ret:
        print("エラー: 最初の背景画像をキャプチャできませんでした。")
        cap.release()
        sys.exit(1)
    
    global initial_far_background
    initial_far_background = cv2.cvtColor(bg_frame_initial, cv2.COLOR_BGR2GRAY)
    initial_far_background = cv2.GaussianBlur(initial_far_background, (31, 31), 0)
    print("「遠景」背景画像をキャプチャしました。")
    # ----------------------------------------

    # 背景差分器のインスタンス化 (ここではFixedBackgroundSubtractorを使用)
    # 動的に生成した背景をセットするため、FixedBackgroundSubtractorが適している
    subtractor = FixedBackgroundSubtractor(blur_ksize=(31, 31), threshold_val=50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("エラー: フレームを読み取れませんでした。")
                break
            
            frame_h, frame_w, _ = frame.shape

            # トラックバーの値を取得し、スケールファクターを計算
            trackbar_pos = cv2.getTrackbarPos(trackbar_name, window_name)
            # trackbar_pos (0-100) をスケールファクター (1.0 - SCALE_FACTOR_MAX) に変換
            # 0 -> 1.0 (等倍), 100 -> SCALE_FACTOR_MAX
            current_scale_factor = 1.0 + (trackbar_pos / 100.0) * (SCALE_FACTOR_MAX - 1.0)

            # --- 動的背景の生成 ---
            # 最初の「遠景」背景画像から、現在のスケールファクターに合わせてクロップ＆リサイズ
            
            # クロップする領域のサイズを計算
            # 例: スケールファクターが2.0なら、元の画像の半分のサイズをクロップ
            crop_w = int(initial_far_background.shape[1] / current_scale_factor)
            crop_h = int(initial_far_background.shape[0] / current_scale_factor)

            # クロップ領域の中心座標
            center_x = initial_far_background.shape[1] // 2
            center_y = initial_far_background.shape[0] // 2

            # クロップ領域の左上座標
            x1 = max(0, center_x - crop_w // 2)
            y1 = max(0, center_y - crop_h // 2)
            x2 = min(initial_far_background.shape[1], x1 + crop_w)
            y2 = min(initial_far_background.shape[0], y1 + crop_h)
            
            # クロップ領域がフレームサイズと一致しない場合の調整
            # (例: crop_wが奇数の場合など)
            crop_w = x2 - x1
            crop_h = y2 - y1

            # クロップ
            cropped_bg = initial_far_background[y1:y2, x1:x2]

            # リサイズして現在のフレームサイズに合わせる
            # これが動的に生成された背景となる
            dynamic_background_gray = cv2.resize(cropped_bg, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            
            # --- 動的背景を背景差分器にセット ---
            subtractor.set_background(cv2.cvtColor(dynamic_background_gray, cv2.COLOR_GRAY2BGR)) # BGRに変換して渡す
            
            # --- 前景マスクの取得 ---
            thresh = subtractor.get_foreground_mask(frame)
            # -----------------------
            
            # --- 輪郭検出と描画 ---
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            found_motion = False
            for contour in contours:
                if cv2.contourArea(contour) < 1000:
                    continue
                found_motion = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if found_motion:
                # print("動きを検知しました！") # 頻繁に出力されるのでコメントアウト
                pass
            # -----------------------
            
            # --- 映像の表示 ---
            cv2.imshow(window_name, frame)
            cv2.imshow("Dynamic Background", dynamic_background_gray) # 生成された背景も表示
            cv2.imshow("Foreground Mask", thresh) # 前景マスクも表示
            # -----------------

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 27はESCキー
                if key == ord('q'):
                    print("'q'キーが押されたため、プログラムを終了します。")
                elif key == 27:
                    print("'ESC'キーが押されたため、プログラムを終了します。")
                break

    finally:
        print("後処理: カメラを解放し、すべてのウィンドウを閉じます。")
        cap.release()
        cv2.destroyAllWindows()
        print("--- プログラム終了 ---")

if __name__ == "__main__":
    main()
