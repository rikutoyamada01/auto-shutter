# 自動距離調整機能（物理的）の実装に関する技術ガイド

このドキュメントでは、`README.md`に記載されている「自動ズーム機能（距離調整）」、すなわち**カメラ自体の物理的な前後移動による距離調整機能**を実装するための技術的なアプローチについて説明します。

この機能の目的は、検出した物体がフレームの左右の端から一定の「余白」を保つように、カメラを載せた台などをモーターで前後に動かすことです。

## 1. 設計思想の再確認

`README.md`で定義されているロジックは以下の通りです。

1.  **目標余白の設定**: フレームの左右の端と、検出された物体の間に確保したい余白の幅をピクセル単位で決めます。（例: `TARGET_MARGIN = 100` ピクセル）
2.  **現在の余白を計算**: リアルタイムの映像から検出した物体のバウンディングボックスを取得し、フレームの端からどれだけ離れているか（現在の余白）を計算します。
3.  **比較と判断**:
    *   **現在の余白 < 目標余白**: 物体がフレームの端に近すぎる（＝大きすぎる）。カメラを**後退**させる必要があります。
    *   **現在の余白 > 目標余白**: 物体がフレーム中央に寄りすぎている（＝小さすぎる）。カメラを**前進**させる必要があります。
    *   **ちょうど良い範囲**: 停止します。

まずは、このロジックを`print`文で確認し、その後ハードウェア（モーター）制御へと進むのが安全な開発ステップです。

## 2. 実装ステップ

`src/main.py`の`while`ループ内に、以下のロジックを組み込んでいきます。

### ステップ 2.1: パラメータの定義

判断の基準となる値を、ループの前に定義しておくと管理がしやすくなります。

```python
def main():
    # ... (既存のセットアップコード)

    # --- 自動距離調整用のパラメータ ---
    # 目標とする左右の余白 (ピクセル単位)
    TARGET_MARGIN = 150 
    # どのくらいの誤差を許容するか (この範囲内なら停止)
    TOLERANCE = 25
    # --------------------------------

    window_name = "Camera Feed"
    # ... (既存のコード)

    try:
        while True:
            # ... (フレーム読み取り)
            frame_h, frame_w, _ = frame.shape # フレームサイズを取得

            # ... (背景差分、輪郭検出)
```

### ステップ 2.2: ターゲットの特定と現在の余白の計算

複数の物体が検出された場合、どれを基準に動くかを決める必要があります。ここでは、最も大きい物体をターゲットとします。

```python
            # --- 輪郭検出と描画 ---
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # デフォルトのモーター指示は「停止」
            motor_command = "停止"

            if contours:
                # 最も面積の大きい輪郭をメインターゲットとする
                main_target = max(contours, key=cv2.contourArea)

                if cv2.contourArea(main_target) > 1000: # ある程度の大きさがある場合のみ処理
                    (x, y, w, h) = cv2.boundingRect(main_target)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # --- ここからが距離調整ロジック ---
                    # 現在の左側の余白
                    current_left_margin = x
                    # 現在の右側の余白
                    current_right_margin = frame_w - (x + w)

                    # より小さい方の余白を基準に判断する
                    current_margin = min(current_left_margin, current_right_margin)

                    # --- 判断ロジック ---
                    if current_margin < TARGET_MARGIN - TOLERANCE:
                        # 余白が目標より狭すぎる -> 物体が大きすぎる
                        motor_command = "後退"
                    elif current_margin > TARGET_MARGIN + TOLERANCE:
                        # 余白が目標より広すぎる -> 物体が小さすぎる
                        motor_command = "前進"
                    else:
                        # 許容範囲内
                        motor_command = "停止"
            
            # 判断結果をコンソールに出力
            print(f"モーターへの指示: {motor_command}")
            # -----------------------
```

### ステップ 2.3: 画面への情報表示（デバッグ用）

現在の状態を分かりやすくするため、計算した余白や目標余白を画面に描画すると、デバッグが非常にやりやすくなります。

```python
            # ... (判断ロジックの後)

            # --- デバッグ情報の描画 ---
            # 目標の余白を示すラインを左右に描画
            cv2.line(frame, (TARGET_MARGIN, 0), (TARGET_MARGIN, frame_h), (255, 0, 0), 2)
            cv2.line(frame, (frame_w - TARGET_MARGIN, 0), (frame_w - TARGET_MARGIN, frame_h), (255, 0, 0), 2)

            # 現在の余白情報をテキストで表示
            info_text = f"Margin: {current_margin}px | Command: {motor_command}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # -----------------------

            # --- 映像の表示 ---
            cv2.imshow(window_name, frame)
```
※ `current_margin` は `if contours:` ブロックの外では未定義になる可能性があるので、テキスト表示はブロック内で行うか、未検出の場合の表示を工夫する必要があります。

## 3. `src/main.py` への統合案

上記のロジックを `src/main.py` に統合した全体のコードイメージです。

```python
# (import文など)

def main():
    # ... (カメラ初期化など)

    # --- 自動距離調整用のパラメータ ---
    TARGET_MARGIN = 150 
    TOLERANCE = 25
    # --------------------------------

    # ... (背景初期化など)

    window_name = "Camera Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_h, frame_w, _ = frame.shape
            thresh = subtractor.get_foreground_mask(frame)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motor_command = "停止"
            found_motion = False

            if contours:
                main_target = max(contours, key=cv2.contourArea)
                if cv2.contourArea(main_target) > 1000:
                    found_motion = True
                    (x, y, w, h) = cv2.boundingRect(main_target)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    current_margin = min(x, frame_w - (x + w))

                    if current_margin < TARGET_MARGIN - TOLERANCE:
                        motor_command = "後退"
                    elif current_margin > TARGET_MARGIN + TOLERANCE:
                        motor_command = "前進"
                    
                    # デバッグ情報を描画
                    info_text = f"Margin: {current_margin}px"
                    cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if found_motion:
                print(f"動きを検知しました！ モーターへの指示: {motor_command}")
            
            # 目標の余白ラインを描画
            cv2.line(frame, (TARGET_MARGIN, 0), (TARGET_MARGIN, frame_h), (255, 0, 0), 2)
            cv2.line(frame, (frame_w - TARGET_MARGIN, 0), (frame_w - TARGET_MARGIN, frame_h), (255, 0, 0), 2)
            cv2.putText(frame, f"Command: {motor_command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        # ... (後処理)

if __name__ == "__main__":
    main()
```

これで、まずはコンソールにモーターへの指示が出力されるようになります。この出力が意図通りであることを確認した上で、次のステップであるハードウェア制御に進んでください。