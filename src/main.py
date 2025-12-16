import cv2
import sys
import time
import math
from enum import Enum, auto
from dataclasses import dataclass
from ultralytics import YOLO # type: ignore

from measure_distance import detect_person_distance2sideedge
from detect_circle_gesture import detect_circle_gesture

# --- 設定値管理 ---
@dataclass(frozen=True)
class Config:
    CAMERA_INDEX: int = 0
    MARGIN: int = 50
    MAX_PICTURE: int = 3
    FPS: int = 30  # FPSを30に設定（処理負荷軽減のため）
    RESOLUTION_WIDTH: int = 640
    RESOLUTION_HEIGHT: int = 480
    
    # 時間設定 (秒)
    ADJUST_DURATION_SEC: float = 5.0      # 調整完了までの時間
    COOLDOWN_DURATION_SEC: float = 2.0    # 撮影後のクールダウン
    COUNTDOWN_SEC: float = 3.0            # ジェスチャー検知から撮影までの秒数
    TAKE_PICTURE_TIMEOUT_SEC: float = 30.0 # 撮影待機が長すぎた場合のタイムアウト
    RESULT_DURATION_SEC: float = 10.0     # 結果表示時間
    
    # フレーム数換算 (初期化時に計算)
    @property
    def ADJUST_FRAMES(self): return int(self.ADJUST_DURATION_SEC * self.FPS)
    @property
    def COOLDOWN_FRAMES(self): return int(self.COOLDOWN_DURATION_SEC * self.FPS)
    @property
    def COUNTDOWN_FRAMES(self): return int(self.COUNTDOWN_SEC * self.FPS)
    @property
    def TAKE_PICTURE_TIMEOUT_FRAMES(self): return int(self.TAKE_PICTURE_TIMEOUT_SEC * self.FPS)
    @property
    def RESULT_FRAMES(self): return int(self.RESULT_DURATION_SEC * self.FPS)
    
    # カメラ設定
    EXPOSURE_VAL: int = 80
    WARMUP_FRAMES: int = 30
    WINDOW_NAME: str = "Photo Booth App"

# --- 状態定義 ---
class AppState(Enum):
    READY = auto()
    ADJUST = auto()
    TAKE_PICTURE = auto()
    PICTURE_COOLDOWN = auto()
    RESULT = auto()

class PhotoBoothApp:
    def __init__(self):
        self.state = AppState.READY
        self.cap = None
        self.subtractor = None
        self.pose_model = None
        self.config = Config() # プロパティアクセス用
        
        # 状態管理用変数
        self.state_timer = 0
        self.taken_pictures_count = 0
        
        # 撮影カウントダウン用
        self.is_counting_down = False
        self.countdown_timer = 0
        # ジェスチャー検出結果のキャッシュ
        self.last_gesture_detected = False
        self.last_frame_with_pose = None

    def initialize(self):
        """カメラとAIモデルの初期化"""
        print("--- システム初期化中 ---")
        
        # YOLOモデルのロード
        print("AIモデルをロード中...")
        self.pose_model = YOLO("yolo11n-pose.pt")

        # カメラセットアップ
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)

        if not self.cap.isOpened():
            print(f"エラー: カメラ(インデックス: {self.config.CAMERA_INDEX})を開けませんでした。")
            # もしRaspberry Piなら、取り付けてあるカメラを使う。
            if self._is_raspberry_pi():
                print("Raspberry Piなので指定のカメラを使います")
                success = self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V")) # type: ignore カメラの機種によって変える
                if success == False:
                    sys.exit(1)
            else:
                sys.exit(1)

            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.RESOLUTION_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.RESOLUTION_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 自動露出OFF (環境による)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.EXPOSURE_VAL)

        # 背景差分の初期化


        # ウォームアップ
        print("カメラ起動中...")
        for _ in range(self.config.WARMUP_FRAMES):
            self.cap.read()

        cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("初期化完了。システムを開始します。")

    def run(self):
        """メインループ"""
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.read_latest(self.cap)
                if not ret:
                    print("フレームの読み込みに失敗")
                    continue

                # 鏡のように左右反転（UX向上のため）
                frame = cv2.flip(frame, 1)

                # 現在の状態に応じた処理を実行
                # process_state内でframeに描画(上書き)を行う
                self._process_state(frame)

                # UI情報のオーバーレイ描画
                self._draw_ui(frame)

                cv2.imshow(self.config.WINDOW_NAME, frame)

                # 入力処理
                if not self._handle_input():
                    break
                
                # FPS制御
                elapsed = time.time() - start_time
                sleep_time = (1.0 / self.config.FPS) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # 入力処理 (waitKeyはキー入力のみに利用し、待機時間は最小限にする)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self._cleanup()

    def _process_state(self, frame):
        """状態ごとのロジック分岐"""
        if self.state == AppState.READY:
            self._handle_ready(frame)
        elif self.state == AppState.ADJUST:
            self._handle_adjust(frame)
        elif self.state == AppState.TAKE_PICTURE:
            self._handle_take_picture(frame)
        elif self.state == AppState.PICTURE_COOLDOWN:
            self._handle_cooldown(frame)
        elif self.state == AppState.RESULT:
            self._handle_result(frame)

    # --- 各状態のハンドラ ---

    def _handle_ready(self, frame):
        """READY: 丸ジェスチャーを待機"""
        # 5フレームに1回だけ推論
        if self.state_timer % 5 == 0: 
            self.last_frame_with_pose, self.last_gesture_detected = detect_circle_gesture(frame)
        
        # 描画結果を反映 (キャッシュから)
        # キャッシュされたフレームがない場合（最初の数フレームなど）は現在のフレームを使用
        frame[:] = self.last_frame_with_pose if self.last_frame_with_pose is not None else frame.copy()

        if self.last_gesture_detected:
            cv2.putText(frame, "STARTING!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            # 即時遷移せず、少しユーザーにフィードバックを見せたい場合はここで少し待つ処理を入れても良い
            # 今回は即座に遷移
            self._transition_to(AppState.ADJUST)
        else:
            cv2.putText(frame, "Make a Circle to Start", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    def _handle_adjust(self, frame):
        """ADJUST: 位置調整"""
        # 距離・位置判定
        is_at_edge = False
        processed_frame = frame 
        # ---------------------------------------------

        try:
            # 距離・位置判定
            # もし detect_person... が外部ファイルになくても止まらないようにtryで囲むのが安全です
            result = detect_person_distance2sideedge(frame, self.config.MARGIN)
            
            # 戻り値が正しく2つあるか確認してから代入
            if result is not None and len(result) == 2:
                processed_frame, is_at_edge = result
                
        except Exception as e:
            print(f"Warning: Distance detection skipped due to error: {e}")

        frame[:] = processed_frame[:] # 描画反映
        
        if is_at_edge:
            cv2.putText(frame, "TOO CLOSE TO EDGE!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # ここでロボット制御などを入れるなら実装
        
        self.state_timer += 1
        
        # プログレスバー風表示 (Result画面に合わせて下部に全幅で表示)
        h, w = frame.shape[:2]
        progress = self.state_timer / self.config.ADJUST_FRAMES
        # 下部20pxのバー
        cv2.rectangle(frame, (0, h-20), (int(w * progress), h), (0, 255, 0), -1)
        
        cv2.putText(frame, "Adjusting...", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.state_timer > self.config.ADJUST_FRAMES:
            self._transition_to(AppState.TAKE_PICTURE)

    def _handle_take_picture(self, frame):
        """TAKE_PICTURE: ジェスチャーでカウントダウン開始 -> 撮影"""
        
        # 1. タイムアウト処理 (操作がない場合、READYに戻る)
        self.state_timer += 1
        if self.state_timer > self.config.TAKE_PICTURE_TIMEOUT_FRAMES and not self.is_counting_down:
            print("タイムアウト: 操作がありませんでした。")
            self._transition_to(AppState.READY)
            return

        # 2. カウントダウン中かどうかで分岐
        if self.is_counting_down:
            self.countdown_timer -= 1
            
            # 残り秒数の計算と表示
            remaining_sec = math.ceil(self.countdown_timer / self.config.FPS)
            
            # 画面中央に大きくカウントダウン表示
            h, w = frame.shape[:2]
            cv2.putText(frame, str(remaining_sec), (w//2 - 50, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
            
            if self.countdown_timer <= 0:
                self._perform_capture(frame)
        else:
            # 3. ジェスチャー待ち
            # 5フレームに1回だけ推論
            if self.state_timer % 5 == 0: 
                self.last_frame_with_pose, self.last_gesture_detected = detect_circle_gesture(frame)
            
            # 描画結果を反映 (キャッシュから)
            # キャッシュされたフレームがない場合（最初の数フレームなど）は現在のフレームを使用
            frame[:] = self.last_frame_with_pose if self.last_frame_with_pose is not None else frame.copy()
            
            cv2.putText(frame, f"Pose for Picture! ({self.taken_pictures_count + 1}/{self.config.MAX_PICTURE})", 
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, "Make Circle to Snap", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            if self.last_gesture_detected:
                print("撮影ジェスチャー検知: カウントダウン開始")
                self.is_counting_down = True
                self.countdown_timer = self.config.COUNTDOWN_FRAMES

    def _perform_capture(self, frame):
        """撮影実行処理"""
        # シャッターエフェクト（画面を白くするなど）を入れると良い
        print("パシャッ！ (撮影)")
        
        self.taken_pictures_count += 1
        
        if self.taken_pictures_count >= self.config.MAX_PICTURE:
            self._transition_to(AppState.RESULT)
        else:
            self._transition_to(AppState.PICTURE_COOLDOWN)

    def _handle_cooldown(self, frame):
        """PICTURE_COOLDOWN: 連続撮影防止と確認用"""
        self.state_timer += 1
        self._shutter_flash(frame, self.state_timer)
        cv2.putText(frame, "Nice Shot!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
        if self.state_timer > self.config.COOLDOWN_FRAMES:
            self._transition_to(AppState.TAKE_PICTURE)

    def _handle_result(self, frame):
        """RESULT: QRコード表示など。時間経過でREADYへ"""
        self.state_timer += 1
        
        # ここにQRコード画像のオーバーレイ処理などを記述
        cv2.putText(frame, "ALL DONE!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(frame, "Thank you for using.", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 残り時間のバー
        remaining_ratio = 1.0 - (self.state_timer / self.config.RESULT_FRAMES)
        cv2.rectangle(frame, (0, frame.shape[0]-20), (int(frame.shape[1] * remaining_ratio), frame.shape[0]), (0, 100, 255), -1)

        if self.state_timer > self.config.RESULT_FRAMES:
            self._transition_to(AppState.READY)

    def _transition_to(self, new_state):
        print(f"Phase Change: {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.state_timer = 0
        self.is_counting_down = False # 状態遷移時にカウントダウンはリセット
        
        # 状態遷移時にジェスチャーキャッシュをリセット
        # これをしないと、前の状態の「検出済み」フラグが残ってしまい
        # 次の状態で即座に反応してしまう可能性がある
        self.last_gesture_detected = False
        self.last_frame_with_pose = None
        
        if new_state == AppState.READY:
             self.taken_pictures_count = 0

    def _draw_ui(self, frame):
        cv2.putText(frame, f"Phase: {self.state.name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # タイムアウトまでの残り時間表示 (TAKE_PICTUREのみ)
        if self.state == AppState.TAKE_PICTURE and not self.is_counting_down:
            remaining = int((self.config.TAKE_PICTURE_TIMEOUT_FRAMES - self.state_timer) / self.config.FPS)
            cv2.putText(frame, f"Timeout: {remaining}s", (frame.shape[1]-200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _handle_input(self) -> bool:
        return True # waitKeyはrunメソッド内で処理済みなのでここは常にTrue

    def _cleanup(self):
        print("後処理を実行します...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("終了")

    def _is_raspberry_pi(self) -> bool:
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
            return "raspberry pi" in model
        except FileNotFoundError:
            return False
        
    def _shutter_flash(self, frame, t, duration=60):
        """
        t: シャッター開始からの経過秒
        """
        progress = min(t / duration, 1.0)
        alpha = 1.0 - progress  # 徐々に消える

        self._shutter_flash_rect(frame, alpha)

    def _shutter_flash_rect(self, frame, alpha=1.0):
        if alpha > 0.01: # alphaが十分に大きい場合のみ実行
            h, w = frame.shape[:2]

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (w, h),
                (255, 255, 255),
                thickness=-1
            )

            frame[:] = cv2.addWeighted(
                overlay,
                alpha,
                frame,
                1 - alpha,
                0
            )

    def read_latest(self, cap):
        return cap.read()



if __name__ == "__main__":
    app = PhotoBoothApp()
    app.initialize()
    app.run()
