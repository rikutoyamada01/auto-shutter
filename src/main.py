# -*- coding: utf-8 -*-
import cv2
import sys

def main():
    """
    カメラからの映像をリアルタイムで表示するメイン関数。
    """
    print("--- カメラテストプログラム開始 ---")
    
    # カメラデバイスを開きます (0は通常、システムのデフォルトカメラ)。
    # カメラが認識されない場合、1, 2, ... と数値を変更してみてください。
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    # カメラが正常に開かれたかを確認します。
    if not cap.isOpened():
        print(f"エラー: カメラ(インデックス: {camera_index})を開けませんでした。")
        print("PCにカメラが接続されているか、他のアプリで使用中でないか確認してください。")
        sys.exit(1)

    print(f"カメラ(インデックス: {camera_index})を正常に開きました。")
    
    window_name = "Camera Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(f"映像ウィンドウ '{window_name}' を表示します。")
    print("ウィンドウを選択した状態で 'q'キー を押すと終了します。")

    try:
        while True:
            # カメラからフレームを1枚読み込みます。
            # retは読み込みが成功したか(True/False)、frameは画像データです。
            ret, frame = cap.read()

            # フレームが正しく読み込めなかった場合(retがFalse)、ループを抜けます。
            if not ret:
                print("エラー: フレームを読み取れませんでした。カメラとの接続が切れた可能性があります。")
                break
            
            # 取得したフレームをウィンドウに表示します。
            cv2.imshow(window_name, frame)

            # 'q'キーが押されたらループを抜けます。
            # waitKey(1)は1ミリ秒キー入力を待ちます。これがないと映像が更新されません。
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q'キーが押されたため、プログラムを終了します。")
                break

    # --- ループ終了後の後処理 ---
    print("後処理: カメラを解放し、すべてのウィンドウを閉じます。")
    cap.release()
    cv2.destroyAllWindows()
    print("--- プログラム終了 ---")

if __name__ == "__main__":
    main()
