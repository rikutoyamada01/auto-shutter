import cv2
import sys
import time
import math
import os
import numpy as np
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass

from detect_circle_gesture import CircleGestureDetector
from profiler import profiler
import simple_server_qr

# --- 設定値管理 ---
@dataclass(frozen=True)
class Config:
    CAMERA_INDEX: int = 0
    MARGIN: int = 50
    MAX_PICTURE: int = 3
    FPS: int = 15  # FPSを15に設定（処理負荷軽減のため）
    RESOLUTION_WIDTH: int = 640
    RESOLUTION_HEIGHT: int = 480
    
    # 時間設定 (秒)
    ADJUST_DURATION_SEC: float = 5.0      # 調整完了までの時間
    COOLDOWN_DURATION_SEC: float = 2.0    # 撮影後のクールダウン
    COUNTDOWN_SEC: float = 3.0            # ジェスチャー検知から撮影までの秒数
    TAKE_PICTURE_TIMEOUT_SEC: float = 30.0 # 撮影待機が長すぎた場合のタイムアウト
    RESULT_DURATION_SEC: float = 20.0     # 結果表示時間
    
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
        self.gesture_detector: CircleGestureDetector | None = None
        self.robot = None
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
        
        # Adjust状態のキャッシュ
        self.last_adjust_frame = None
        self.last_is_at_edge = False
        
        self.is_pi = self._check_is_raspberry_pi()
        print(f"[DEBUG] Device is Raspberry Pi: {self.is_pi}")

        # QR Server & Photo Storage
        self.captured_files = []
        self.server_needs_update = False
        self.qr_server_stop = None
        self.qr_image_cv = None
        self.qr_urls = []
        self.output_dir = "captured_photos"
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize(self):
        """カメラとAIモデルの初期化"""
        print("--- システム初期化中 ---")
        
        # MediaPipeジェスチャー検知器の初期化
        print("MediaPipeモデルをロード中...")
        self.gesture_detector = CircleGestureDetector()

        # カメラセットアップ
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)

        if not self.cap.isOpened():
            print(f"エラー: カメラ(インデックス: {self.config.CAMERA_INDEX})を開けませんでした。")
            # もしRaspberry Piなら、取り付けてあるカメラを使い、モーターを動かす。
            if self.is_pi:
                print("Raspberry Piなので指定のカメラを使います")
                success = self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V")) # type: ignore カメラの機種によって変える
                if success == False:
                    sys.exit(1)
            else:
                sys.exit(1)

        # Raspberry Piならロボット（モーター）を初期化する
        if self.is_pi:
            try:
                from gpiozero import Robot, Motor
                import gpiozero
                print(f"[DEBUG] Using gpiozero pin factory: {gpiozero.Device.pin_factory}")
                print("[DEBUG] Initializing Robot (GPIO 17,18,19,20)...")
                
                # PWMを有効化(デフォルト)に戻し、速度制御できるようにする
                # 引数を指定しなければデフォルトでpwm=Trueになります
                self.robot = Robot(left=(17,18), right=(19,20))
                
                # 構成情報の詳細表示
                print(f"[DEBUG] Robot Object: {self.robot}")
                print(f"[DEBUG] Left Motor: {self.robot.left_motor}")
                print(f"[DEBUG] Right Motor: {self.robot.right_motor}")
                
                # 接続テスト: 一瞬だけ動かしてみる(速度0.4)
                print("[DEBUG] Performing Motor Self-Test (0.1s backward at speed 0.4)...")
                self.robot.backward(speed=0.4) # type: ignore
                time.sleep(0.1)
                self.robot.stop()
                print("[DEBUG] Motor Self-Test Complete.")
            except Exception as e:
                print(f"Warning: Motor initialization or test failed: {e}")

            
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
                

                with profiler.measure("cap_read"):
                    ret, frame = self.read_latest(self.cap)
                if not ret:
                    print("フレームの読み込みに失敗")
                    continue

                # 鏡のように左右反転（UX向上のため）
                with profiler.measure("cv2_flip"):
                    frame = cv2.flip(frame, 1)

                # 現在の状態に応じた処理を実行
                # process_state内でframeに描画(上書き)を行う
                with profiler.measure(f"process_state_{self.state.name}"):
                    self._process_state(frame)

                # UI情報のオーバーレイ描画
                with profiler.measure("draw_ui"):
                    self._draw_ui(frame)

                with profiler.measure("imshow"):
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
        if self.state_timer % 5 == 0 and self.gesture_detector: 
            with profiler.measure("detect_circle_gesture"):
                # MediaPipe detector call
                self.last_frame_with_pose, self.last_gesture_detected = self.gesture_detector.detect(frame)
        
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
        
        self.state_timer += 1

    def _handle_adjust(self, frame):
        """ADJUST: 位置調整"""
        # 距離・位置判定
        is_at_edge = False
        processed_frame = frame 
        # ---------------------------------------------

        try:
            # 距離・位置判定 (5フレームに1回)
            if self.state_timer % 5 == 0 and self.gesture_detector:
                with profiler.measure("detect_edge_proximity"):
                    # Use the same gesture detector for edge detection (distance check)
                    processed_frame_res, is_at_edge_res = self.gesture_detector.detect_edge_proximity(
                        frame.copy(), self.config.MARGIN
                    )
                    
                    # Ensure the frame from MediaPipe is valid before using it
                    if processed_frame_res is not None and isinstance(processed_frame_res, np.ndarray) and processed_frame_res.ndim >= 2:
                        self.last_adjust_frame = processed_frame_res
                        self.last_is_at_edge = is_at_edge_res
                    else:
                        print("Warning: Invalid frame received from MediaPipe detector_edge_proximity.")
                        self.last_adjust_frame = None
                        self.last_is_at_edge = False
            
            # キャッシュを使用
            if self.last_adjust_frame is not None:
                processed_frame = self.last_adjust_frame
                is_at_edge = self.last_is_at_edge
            else:
                # 初回などキャッシュがない場合
                processed_frame = frame
                is_at_edge = False

        except Exception as e:
            print(f"Warning: Distance detection skipped due to error: {e}")
            processed_frame = frame
            is_at_edge = False

        frame[:] = processed_frame[:] # 描画反映
        
        if is_at_edge:
            cv2.putText(frame, "TOO CLOSE TO EDGE!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if self.is_pi and self.robot:
                # 後退する(ロボット制御)
                try:
                    print("[DEBUG] Robot BACKWARD (Speed: 0.4)")
                    self.robot.backward(speed=0.4) # type: ignore
                except Exception as e:
                    print(f"Warning: Robot is not moving backward: {e}")
        else:
            if self.is_pi and self.robot:
                try:
                    # print("[DEBUG] Robot STOP") # ログ過多防止のためコメントアウト。必要なら解除
                    self.robot.stop()
                except:
                    pass
                

        
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
            if self.state_timer % 5 == 0 and self.gesture_detector: 
                with profiler.measure("detect_circle_gesture"):
                    self.last_frame_with_pose, self.last_gesture_detected = self.gesture_detector.detect(frame)
            
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
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}_{self.taken_pictures_count + 1}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved: {filepath}")
        self.captured_files.append(filepath)
        
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
        
        # Start server if needed (update with new files)
        if self.server_needs_update and self.captured_files:
            # If server is already running, stop it first
            if self.qr_server_stop:
                print("Stopping previous QR Server...")
                self.qr_server_stop()
                self.qr_server_stop = None

            print("Starting QR Server...")
            try:
                # Provide absolute paths to avoid directory issues
                abs_files = [os.path.abspath(p) for p in self.captured_files]
                qr_pil, urls, stop_func = simple_server_qr.serve_and_generate_qr(abs_files)
                self.qr_server_stop = stop_func
                self.qr_urls = urls
                
                # Convert PIL to OpenCV (RGB -> BGR)
                # Ensure it's RGB first (qrcode often returns 1-bit '1' mode)
                self.qr_image_cv = np.array(qr_pil.convert('RGB'))
                self.qr_image_cv = cv2.cvtColor(self.qr_image_cv, cv2.COLOR_RGB2BGR)
                print(f"Server started at {urls[0]}")
            except Exception as e:
                print(f"Failed to start QR server: {e}")
            
            self.server_needs_update = False

        self.state_timer += 1
        
        # Draw QR Code if available
        if self.qr_image_cv is not None:
            # Resize QR to fit nicely (e.g., 50% of screen height)
            qr_h, qr_w = self.qr_image_cv.shape[:2]
            target_h = int(frame.shape[0] * 0.5)
            if qr_h > 0:
                scale = target_h / qr_h
                resized_qr = cv2.resize(self.qr_image_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                
                # Center overlay
                rh, rw = resized_qr.shape[:2]
                y_offset = (frame.shape[0] - rh) // 2
                x_offset = (frame.shape[1] - rw) // 2
                
                # Check bounds
                if y_offset >= 0 and x_offset >= 0 and y_offset+rh <= frame.shape[0] and x_offset+rw <= frame.shape[1]:
                    frame[y_offset:y_offset+rh, x_offset:x_offset+rw] = resized_qr
                
                # Draw URL text above QR
                if self.qr_urls:
                    text = "Scan to Download!"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    text_x = (frame.shape[1] - tw) // 2
                    cv2.putText(frame, text, (text_x, y_offset - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
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
        
        if new_state == AppState.RESULT:
            self.server_needs_update = True
        
        # Reset captured files for next session if leaving RESULT state
        # The server will continue running with the old files until updated in _handle_result
        if self.state == AppState.RESULT and new_state != AppState.RESULT:
            self.captured_files = []

        # 状態遷移時にロボットを停止させる
        if self.robot:
            try:
                self.robot.stop()
            except:
                pass

        self.state = new_state
        self.state_timer = 0
        self.is_counting_down = False # 状態遷移時にカウントダウンはリセット
        
        # 状態遷移時にジェスチャーキャッシュをリセット
        # これをしないと、前の状態の「検出済み」フラグが残ってしまい
        # 次の状態で即座に反応してしまう可能性がある
        self.last_gesture_detected = False
        self.last_frame_with_pose = None
        self.last_adjust_frame = None
        self.last_is_at_edge = False
        
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
        if self.qr_server_stop:
            self.qr_server_stop()
        if self.robot:
            try:
                self.robot.stop()
            except:
                pass
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("終了")

    def _check_is_raspberry_pi(self) -> bool:
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
            is_pi = "raspberry pi" in model
            print(f"[DEBUG] check_is_raspberry_pi: {is_pi} (model: {model.strip()})")
            return is_pi
        except FileNotFoundError:
            print("[DEBUG] check_is_raspberry_pi: False (File not found)")
            return False
        
    def _shutter_flash(self, frame, t, duration=30):
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
