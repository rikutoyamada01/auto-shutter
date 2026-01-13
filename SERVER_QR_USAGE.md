
# 新機能: GUI 写真共有ツール & 再利用可能モジュール

このドキュメントでは、新たに追加された写真共有用GUIアプリと、そのバックエンドとして機能する再利用可能なモジュールについて説明します。

## 1. GUI 写真共有ツール (`gui_app.py`)

パソコン内の写真を簡単にローカルネットワークで共有するためのグラフィカルなツールです。

### 使い方

1.  **起動方法**:
    仮想環境を有効にした状態で、以下のコマンドを実行します。
    ```powershell
    python gui_app.py
    ```

2.  **操作手順**:
    -   ウィンドウが表示されたら、「**Select Photos**」ボタンをクリックします。
    -   共有したい写真ファイル（複数選択可）を選びます。
    -   自動的に一時的なWebサーバーが起動し、ウィンドウ内に **QRコード** が表示されます。
    -   同じWi-Fiに接続しているスマホでQRコードを読み取ると、共有ページにアクセスして写真をダウンロードできます。

3.  **終了**:
    -   ウィンドウを閉じると、Webサーバーも自動的に停止します。

---

## 2. 再利用可能モジュール (`simple_server_qr.py`)

指定されたファイルをホストする一時的なWebサーバーをバックグラウンドで起動し、そのURLのQRコードを生成するためのPythonモジュールです。独自のツールに組み込んで使用できます。

### 機能
-   **自動IP検出**: `get_local_ip()` でLAN内の適切なIPアドレスを自動取得します。
-   **バックグラウンド実行**: Webサーバー（Flask）を別スレッドで実行し、メインプログラムをブロックしません。
-   **QRコード生成**: 生成されたURLを即座にQRコード画像（PIL Image）として返します。

### 組み込み方（サンプルコード）

```python
import time
import simple_server_qr

# 1. 共有したいファイルのパスリストを作成
files = ["photo1.jpg", "photo2.png"]

# 2. サーバー起動 & QR生成
# qr_img: PIL.Imageオブジェクト
# url: 生成された共有URL
# stop_server: サーバーを停止するための関数
qr_img, url, stop_server = simple_server_qr.serve_and_generate_qr(files)

print(f"Server started at: {url}")
qr_img.show() # QRコードを表示

try:
    # サーバーを維持したい間は待機
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # 3. サーバー停止
    stop_server()
    print("Server stopped.")
```

### コマンドライン実行

簡易的なテストとして、このファイルを直接実行することも可能です。
```powershell
python simple_server_qr.py image1.jpg image2.png
```

---

## 依存環境のセットアップ

これらのツールを使用するには、以下のライブラリが必要です。

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
(`requirements.txt` には `Flask`, `Pillow`, `qrcode` が含まれています)

## トラブルシューティング

**Q. スマホから「接続できません」「タイムアウト」と表示される**
1.  **Windows Firewall**: ファイアウォールがPythonの通信をブロックしている可能性があります。「Windows Defender ファイアウォール」の設定で、`python.exe` の通信を許可するか、一時的に無効にして試してください。
2.  **IPアドレス**: アプリに表示されたURLが正しいか確認してください。複数のIPが表示されている場合は、別のURLを試してください（特に `192.168.x.x` で始まるものが一般的に正解です）。
3.  **ネットワーク分離**: カフェや公共のWi-Fiでは、セキュリティのため機器同士の通信がブロックされていることがあります（AP分離機能）。自宅のWi-Fiやテザリングで試してください。
