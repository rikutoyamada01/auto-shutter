import cv2
from ultralytics import YOLO

def detect_person(source=0):
    """
    YOLO11nを使用して動画から人（全体）を検出する
    source: 動画ファイルのパス または カメラID (例: 0)
    """
    
    # 1. モデルの読み込み
    model = YOLO("yolo11n.pt")

    # 2. 動画ソースを開く
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("エラー: 動画またはカメラを開けませんでした。")
        return

    print("検出を開始します... 'q'キーで終了します。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. 推論を実行
        # classes=[0] を指定することで、「人 (person)」クラスのみを検出します
        # 信頼度(conf)が0.5以上のものだけを表示したい場合は conf=0.5 を追加してください
        results = model(frame, classes=[0], verbose=False)

        # 4. 結果の描画
        # ultralyticsの便利なメソッド plot() を使うと、
        # 枠線、ラベル、信頼度を自動で描画してくれます。
        annotated_frame = results[0].plot()

        # もし自分で色や線の太さをカスタマイズしたい場合は、
        # 前回のコードのように results[0].boxes をループして cv2.rectangle を使ってください。

        # 5. 結果を表示
        cv2.imshow("YOLO11n Person Detection", annotated_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Webカメラの場合は 0、動画ファイルの場合はファイルパスを指定
    detect_person(0)
