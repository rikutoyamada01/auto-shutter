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
from ui_overlay import UIOverlay
from ui_text import UIText
from logger import LoggerManager, LogConfig

# --- 設定値管理 ---
@dataclass(frozen=True)
class Config:
    CAMERA_INDEX: int = 0
    MARGIN: int = 80
    MOTOR_SPEED: float = 0.3 # モーター速度 (0.2では動かない)
    MAX_PICTURE: int = 3
    FPS: int = 15  # FPSを15に設定（処理負荷軽減のため）
    RESOLUTION_WIDTH: int = 640
    RESOLUTION_HEIGHT: int = 480
    
    # 時間設定 (秒)
    ADJUST_DURATION_SEC: float = 5.0      # 調整完了までの時間
    COOLDOWN_DURATION_SEC: float = 2.0    # 撮影後のクールダウン
    COUNTDOWN_SEC: float = 3.0            # ジェスチャー検知から撮影までの秒数
    TAKE_PICTURE_TIMEOUT_SEC: float = 30.0 # 撮影待機が長すぎた場合のタイムアウト
    PRE_ADJUST_DURATION_SEC: float = 2.0  # ADJUST開始前の待機時間
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
    def PRE_ADJUST_FRAMES(self): return int(self.PRE_ADJUST_DURATION_SEC * self.FPS)
    @property
    def RESULT_FRAMES(self): return int(self.RESULT_DURATION_SEC * self.FPS)
    
    # カメラ設定
    EXPOSURE_VAL: int = 80
    WARMUP_FRAMES: int = 30
    WINDOW_NAME: str = "Photo Booth App"

# --- 状態定義 ---
class AppState(Enum):
    READY = auto()
    PRE_ADJUST = auto()
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
        
        # ロガーの初期化
        self.logger = LoggerManager(LogConfig())
        
        # 状態管理用変数
        self.state_timer = 0
        self.taken_pictures_count = 0
        self.frame_count = 0
        
        # 撮影カウントダウン用
        self.is_counting_down = False
        self.countdown_timer = 0
        # ジェスチャー検出結果のキャッシュ
        self.last_gesture_detected = False
        self.last_frame_with_pose = None
        
        # Adjust状態のキャッシュ
        self.last_adjust_frame = None
        self.last_is_at_edge = False
        
        # Camera State
        self.read_failures = 0
        self.MAX_READ_FAILURES = 5
        
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

        # UI Overlay & State
        self.overlay = UIOverlay()
        self.use_japanese = self.overlay.use_japanese
        self.ui_status_text = ""
        self.ui_main_text = ""
        self.ui_sub_text = ""
        self.ui_center_text = ""
        
        # Tracking Log
        self.tracking_log_file = "tracking_log.csv"
        if not os.path.exists(self.tracking_log_file):
            with open(self.tracking_log_file, "w") as f:
                f.write("timestamp,center_error,frame_width\n")
        
        self.last_bbox = None
        self.ui_center_color = (0, 255, 0) # Default Green
        self.ui_progress = None
        self.ui_qr_image = None

    def initialize(self):
        """カメラとAIモデルの初期化"""
        print("--- システム初期化中 ---")
        
        # MediaPipeジェスチャー検知器の初期化
        print("MediaPipeモデルをロード中...")
        self.gesture_detector = CircleGestureDetector()

        # カメラセットアップ
        # カメラセットアップ
        self.cap = self._setup_camera()

        # Robot（モーター）を初期化する (gpiozeroが使える環境なら試みる)
        try:
            from gpiozero import Robot, Motor
            import gpiozero
            print(f"[DEBUG] Using gpiozero pin factory: {gpiozero.Device.pin_factory}")
            print("[DEBUG] Initializing Robot (GPIO 17,18,19,20)...")
            
            # PWMを有効化(デフォルト)に戻し、速度制御できるようにする
            self.robot = Robot(left=(17,18), right=(19,20))
            
            # 構成情報の詳細表示
            print(f"[DEBUG] Robot Object: {self.robot}")
            
            # 接続テスト: 一瞬だけ動かしてみる
            print(f"[DEBUG] Performing Motor Self-Test (0.1s backward at speed {self.config.MOTOR_SPEED})...")
            self.robot.backward(speed=self.config.MOTOR_SPEED) # type: ignore
            time.sleep(0.5)
            self.robot.stop()
            print("[DEBUG] Motor Self-Test Complete.")
            
        except ImportError:
            print("Info: gpiozero module not found. Robot control disabled.")
        except Exception as e:
            print(f"Warning: Motor initialization or test failed: {e}")

    def _setup_camera(self):
        """カメラの初期化と設定"""
        print(f"Connecting to Camera (Index: {self.config.CAMERA_INDEX})...")
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX)

        if not cap.isOpened():
            print(f"Error: Could not open camera (Index: {self.config.CAMERA_INDEX}).")
            if self.is_pi:
                # Raspberry Pi fallback logic if needed, or just fail
                pass
            return cap # Return even if not opened, to be handled by caller if needed, or loop will retry

        if self.is_pi:
            print("Raspberry Pi detected: Setting YUYV format.")
            success = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))
            if not success:
                print("Warning: Failed to set YUYV format.")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.RESOLUTION_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.RESOLUTION_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.config.FPS)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 自動露出OFF (環境による)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, self.config.EXPOSURE_VAL)

        return cap



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
                loop_start_time = time.time()
                self.frame_count += 1
                
                # タイミング計測用
                timings = {}

                with profiler.measure("cap_read"):
                    cap_start = time.time()
                    ret, frame = self.read_latest(self.cap)
                    timings['cap_read'] = (time.time() - cap_start) * 1000
                
                if not ret:
                    self.read_failures += 1
                    print(f"Warning: Failed to read frame ({self.read_failures}/{self.MAX_READ_FAILURES})")
                    time.sleep(0.5)
                    
                    if self.read_failures >= self.MAX_READ_FAILURES:
                        print("Error: Max read failures reached. Attempting to reconnect camera...")
                        if self.cap:
                            self.cap.release()
                        
                        # Re-initialize
                        self.cap = self._setup_camera()
                        self.read_failures = 0
                        time.sleep(1.0) # Wait for camera to come up
                    continue
                
                # Success
                self.read_failures = 0

                # 鏡のように左右反転（UX向上のため）
                with profiler.measure("cv2_flip"):
                    frame = cv2.flip(frame, 1)

                # 現在の状態に応じた処理を実行
                # process_state内でframeに描画(上書き)を行う
                # Reset UI State
                self.ui_status_text = ""
                self.ui_main_text = ""
                self.ui_sub_text = ""
                self.ui_center_text = ""
                self.ui_progress = None
                self.ui_qr_image = None
                
                with profiler.measure(f"process_state_{self.state.name}"):
                    process_start = time.time()
                    self._process_state(frame)
                    timings['process_state'] = (time.time() - process_start) * 1000

                # UI情報のオーバーレイ描画
                with profiler.measure("draw_ui"):
                    draw_start = time.time()
                    self._draw_ui(frame)
                    timings['draw_ui'] = (time.time() - draw_start) * 1000

                with profiler.measure("imshow"):
                    imshow_start = time.time()
                    cv2.imshow(self.config.WINDOW_NAME, frame)
                    timings['imshow'] = (time.time() - imshow_start) * 1000

                # パフォーマンスログを記録
                frame_time = (time.time() - loop_start_time) * 1000
                fps = 1000.0 / frame_time if frame_time > 0 else 0
                self.logger.log_performance(
                    fps=fps,
                    frame_time_ms=frame_time,
                    state=self.state.name,
                    timings=timings
                )

                # 入力処理
                if not self._handle_input():
                    break
                
                # FPS制御
                elapsed = time.time() - loop_start_time
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
        elif self.state == AppState.PRE_ADJUST:
             self._handle_pre_adjust(frame)
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
            
            # ジェスチャーログを記録
            self.logger.log_gesture(
                state=self.state.name,
                detected=self.last_gesture_detected,
                hand_pos=None,  # TODO: 手の位置を取得する場合は追加
                confidence=1.0 if self.last_gesture_detected else 0.0,
                countdown_active=False,
                frame_number=self.frame_count
            )
        
        # 描画結果を反映 (キャッシュから)
        # キャッシュされたフレームがない場合（最初の数フレームなど）は現在のフレームを使用
        frame[:] = self.last_frame_with_pose if self.last_frame_with_pose is not None else frame.copy()

        if self.last_gesture_detected:
            self.ui_center_text = UIText.get_text("starting", self.use_japanese)
            self._transition_to(AppState.PRE_ADJUST)
        else:
            self.ui_main_text = UIText.get_text("ready_gesture", self.use_japanese)
        
        self.state_timer += 1

    def _handle_pre_adjust(self, frame):
        """PRE_ADJUST: ジェスチャー検知後、Adjust開始までの待機 (手を下ろす猶予)"""
        self.state_timer += 1
        
        # 描画はREADYの最後のフレーム(結果表示)を維持するか、あるいはカメラ映像そのままでも良い
        # ここでは普通にカメラ映像を表示しつつ、カウントダウン的なテキストを出す
        self.ui_center_text = UIText.get_text("lower_hands", self.use_japanese)
        self.ui_progress = self.state_timer / self.config.PRE_ADJUST_FRAMES
        
        if self.state_timer > self.config.PRE_ADJUST_FRAMES:
             self._transition_to(AppState.ADJUST)

    def _handle_adjust(self, frame):
        """ADJUST: 位置調整"""
        # 距離・位置判定
        current_edges = set()
        processed_frame = frame 
        # ---------------------------------------------

        try:
            # 距離・位置判定 (5フレームに1回)
            if self.state_timer % 5 == 0 and self.gesture_detector:
                with profiler.measure("detect_edge_proximity"):
                    # Use the same gesture detector for edge detection (distance check)
                    processed_frame_res, edges_res, bbox_res = self.gesture_detector.detect_edge_proximity(
                        frame.copy(), self.config.MARGIN
                    )
                    
                    # Ensure the frame from MediaPipe is valid before using it
                    if processed_frame_res is not None and isinstance(processed_frame_res, np.ndarray) and processed_frame_res.ndim >= 2:
                        self.last_adjust_frame = processed_frame_res
                        self.last_is_at_edge = edges_res # actually a set now
                        self.last_bbox = bbox_res
                    else:
                        print("Warning: Invalid frame received from MediaPipe detector_edge_proximity.")
                        self.last_adjust_frame = None
                        self.last_is_at_edge = set()
                        self.last_bbox = None
            
            # キャッシュを使用
            if self.last_adjust_frame is not None:
                processed_frame = self.last_adjust_frame
                current_edges = self.last_is_at_edge
            else:
                # 初回などキャッシュがない場合
                processed_frame = frame
                current_edges = set()

        except Exception as e:
            print(f"Warning: Distance detection skipped due to error: {e}")
            processed_frame = frame
            current_edges = set()

        frame[:] = processed_frame[:] # 描画反映
        
        # Logic:
        # 1. Top or (Right and Left) -> Backward
        # 2. Right only -> Turn Right (Left Wheel Forward)
        # 3. Left only -> Turn Left (Right Wheel Forward)
        
        is_top = "TOP" in current_edges
        is_left = "LEFT" in current_edges
        is_right = "RIGHT" in current_edges
        is_far = "FAR" in current_edges
        
        # Priority: Edge (Safety/Framing) > Approach (Too Far)
        # needs_backward: Top or (Left AND Right)
        needs_backward = is_top or (is_left and is_right)
        
        # ロボット動作の決定
        robot_action = "NONE"
        motor_speed = 0.0
        
        if current_edges:
            # Visual Feedback
            warnings = [e for e in current_edges if e in ["TOP", "LEFT", "RIGHT"]]
            if warnings:
                # Translation for edges
                translated_warnings = []
                if "TOP" in warnings: translated_warnings.append(UIText.get_text("edge_top", self.use_japanese))
                if "LEFT" in warnings: translated_warnings.append(UIText.get_text("edge_left", self.use_japanese))
                if "RIGHT" in warnings: translated_warnings.append(UIText.get_text("edge_right", self.use_japanese))
                
                prefix = UIText.get_text("too_close", self.use_japanese)
                self.ui_center_text = f"{prefix}: {', '.join(translated_warnings)}"
                self.ui_center_color = (0, 0, 255) # Red
            elif is_far:
                 self.ui_main_text = UIText.get_text("approaching", self.use_japanese)

            if self.robot:
                try:
                    if needs_backward:
                        robot_action = "BACKWARD"
                        motor_speed = self.config.MOTOR_SPEED
                        print(f"[DEBUG] Robot BACKWARD (Speed: {motor_speed})")
                        self.robot.backward(speed=motor_speed)
                    elif is_right:
                        # Person is on RIGHT edge -> Turn RIGHT to center them.
                        robot_action = "TURN_RIGHT"
                        motor_speed = self.config.MOTOR_SPEED
                        print(f"[DEBUG] Robot TURN RIGHT (Left Motor Forward)")
                        self.robot.left_motor.forward(speed=motor_speed)
                        self.robot.right_motor.stop()
                    elif is_left:
                        # Person is on LEFT edge -> Turn LEFT to center them.
                        robot_action = "TURN_LEFT"
                        motor_speed = self.config.MOTOR_SPEED
                        print(f"[DEBUG] Robot TURN LEFT (Right Motor Forward)")
                        self.robot.right_motor.forward(speed=motor_speed)
                        self.robot.left_motor.stop()
                    elif is_far:
                        # APPROACH: Only if NO other edge warnings (Top, Left, Right)
                        robot_action = "FORWARD"
                        motor_speed = self.config.MOTOR_SPEED
                        print(f"[DEBUG] Robot FORWARD (Speed: {motor_speed})")
                        self.robot.forward(speed=motor_speed)
                        
                except Exception as e:
                    print(f"Warning: Robot move failed: {e}")
        else:
            robot_action = "STOP"
            if self.robot:
                try:
                    # print("[DEBUG] Robot STOP")
                    self.robot.stop()
                except:
                    pass
        
        # 距離調整ログを記録（5フレームに1回）
        if self.state_timer % 5 == 0:
            self.logger.log_distance(
                edges=current_edges,
                bbox=self.last_bbox,
                frame_width=self.config.RESOLUTION_WIDTH,
                frame_center=self.config.RESOLUTION_WIDTH / 2,
                robot_action=robot_action,
                motor_speed=motor_speed,
                frame_number=self.frame_count
            )

        
        self.state_timer += 1
        
        # プログレスバー風表示 (Result画面に合わせて下部に全幅で表示)
        self.ui_progress = self.state_timer / self.config.ADJUST_FRAMES
        
        if not current_edges:
             self.ui_main_text = UIText.get_text("adjusting", self.use_japanese)

        if self.state_timer > self.config.ADJUST_FRAMES:
            # Log Tracking Accuracy
            if self.last_bbox:
                min_x, max_x, _, _ = self.last_bbox
                frame_center = self.config.RESOLUTION_WIDTH / 2
                bbox_center = (min_x + max_x) / 2
                error = abs(bbox_center - frame_center)
                
                try:
                    with open(self.tracking_log_file, "a") as f:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{timestamp},{error:.2f},{self.config.RESOLUTION_WIDTH}\n")
                        print(f"[LOG] Tracking Error: {error:.2f}px")
                except Exception as e:
                    print(f"Warning: Failed to log tracking error: {e}")

            self._transition_to(AppState.TAKE_PICTURE)

    def _handle_take_picture(self, frame):
        """TAKE_PICTURE: ジェスチャーでカウントダウン開始 -> 撮影"""
        
        # 1. タイムアウト処理 (操作がない場合、READYに戻る)
        self.state_timer += 1
        
        # Timeout Progress Bar
        if not self.is_counting_down:
            self.ui_progress = 1.0 - (self.state_timer / self.config.TAKE_PICTURE_TIMEOUT_FRAMES)

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
            # 画面中央に大きくカウントダウン表示
            self.ui_center_text = str(remaining_sec)
            
            # カウントダウン中も下部に「ポーズをとって！」を表示
            self.ui_main_text = UIText.get_text("pose", self.use_japanese)
            self.ui_sub_text = f"{self.taken_pictures_count + 1} / {self.config.MAX_PICTURE}"
            
            if self.countdown_timer <= 0:
                self._perform_capture(frame)
        else:
            # 3. ジェスチャー待ち
            # 5フレームに1回だけ推論
            if self.state_timer % 5 == 0 and self.gesture_detector: 
                with profiler.measure("detect_circle_gesture"):
                    self.last_frame_with_pose, self.last_gesture_detected = self.gesture_detector.detect(frame)
                
                # ジェスチャーログを記録
                self.logger.log_gesture(
                    state=self.state.name,
                    detected=self.last_gesture_detected,
                    hand_pos=None,
                    confidence=1.0 if self.last_gesture_detected else 0.0,
                    countdown_active=self.is_counting_down,
                    frame_number=self.frame_count
                )
            
            # 描画結果を反映 (キャッシュから)
            # キャッシュされたフレームがない場合（最初の数フレームなど）は現在のフレームを使用
            frame[:] = self.last_frame_with_pose if self.last_frame_with_pose is not None else frame.copy()
            
            self.ui_main_text = UIText.get_text("take_picture_gesture", self.use_japanese)
            self.ui_sub_text = f"{self.taken_pictures_count + 1} / {self.config.MAX_PICTURE}"

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
        self.ui_center_text = UIText.get_text("photo_taken", self.use_japanese)
        
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
             self.ui_qr_image = self.qr_image_cv
             if self.qr_urls:
                 self.ui_sub_text = self.qr_urls[0]
             
             # Avoid overlap with QR code
             self.ui_center_text = None
             self.ui_main_text = UIText.get_text("complete", self.use_japanese)
        else:
             self.ui_center_text = UIText.get_text("complete", self.use_japanese)
             self.ui_main_text = UIText.get_text("thank_you", self.use_japanese)
        
        # 残り時間のバー
        self.ui_progress = 1.0 - (self.state_timer / self.config.RESULT_FRAMES)

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
             
        # UIステートのリセット
        self.ui_center_text = ""
        self.ui_center_color = (0, 255, 0)
        self.ui_main_text = ""
        self.ui_sub_text = ""
        self.ui_status_text = ""
        self.ui_progress = None

    def _draw_ui(self, frame):
        # 1. Header
        # Phase (Left) + Timeout/Status (Right)
        
        # Override status for Timeout if applicable
        if self.state.name == "TAKE_PICTURE" and not self.is_counting_down:
             remaining = int((self.config.TAKE_PICTURE_TIMEOUT_FRAMES - self.state_timer) / self.config.FPS)
             remaining_text = UIText.get_text("remaining", self.use_japanese)
             sec_text = UIText.get_text("seconds", self.use_japanese)
             self.ui_status_text = f"{remaining_text}: {remaining}{sec_text}"

        # Translate State Name for Header
        state_key = f"state_{self.state.name.lower()}"
        state_text = UIText.get_text(state_key, self.use_japanese)
        status_label = UIText.get_text("status", self.use_japanese)
        self.overlay.draw_header(frame, f"{status_label}: {state_text}", self.ui_status_text)

        # 2. Footer
        self.overlay.draw_footer(frame, self.ui_main_text, self.ui_sub_text, self.ui_progress)

        # 3. Center Status
        if self.ui_center_text:
            self.overlay.draw_center_status(frame, self.ui_center_text, color=self.ui_center_color)
            
        # 4. QR Code (Special Case)
        if self.state == AppState.RESULT and self.ui_qr_image is not None:
             self.overlay.draw_qr_result(frame, self.ui_qr_image)

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
        
        # ロガーを閉じる
        if self.logger:
            self.logger.close()
        
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
