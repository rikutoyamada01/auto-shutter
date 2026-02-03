import os
import csv
import json
import psutil
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class LogConfig:
    """ログ設定"""
    base_dir: str = "logs"
    session_prefix: str = "session"
    enable_performance: bool = True
    enable_gesture: bool = True
    enable_distance: bool = True


class CSVLogger:
    """CSV形式でログを記録する基底クラス"""
    
    def __init__(self, filepath: str, headers: List[str]):
        self.filepath = filepath
        self.headers = headers
        self.file = None
        self.writer = None
        self._initialize()
    
    def _initialize(self):
        """ファイルとCSVライターを初期化"""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
        self.writer.writeheader()
        self.file.flush()
    
    def log(self, data: Dict[str, Any]):
        """データを記録"""
        if self.writer:
            self.writer.writerow(data)
            self.file.flush()
    
    def close(self):
        """ファイルを閉じる"""
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None


class PerformanceLogger(CSVLogger):
    """パフォーマンスログを記録"""
    
    def __init__(self, filepath: str):
        headers = [
            'timestamp',
            'fps',
            'frame_time_ms',
            'state',
            'cap_read_ms',
            'process_state_ms',
            'draw_ui_ms',
            'imshow_ms',
            'memory_mb'
        ]
        super().__init__(filepath, headers)
        self.process = psutil.Process()
    
    def log_frame(self, fps: float, frame_time_ms: float, state: str, 
                  timings: Dict[str, float]):
        """フレーム処理のパフォーマンスを記録
        
        Args:
            fps: 現在のFPS
            frame_time_ms: フレーム処理時間（ミリ秒）
            state: 現在の状態
            timings: 各処理の時間（ミリ秒）の辞書
        """
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'fps': f'{fps:.2f}',
            'frame_time_ms': f'{frame_time_ms:.2f}',
            'state': state,
            'cap_read_ms': f"{timings.get('cap_read', 0):.2f}",
            'process_state_ms': f"{timings.get('process_state', 0):.2f}",
            'draw_ui_ms': f"{timings.get('draw_ui', 0):.2f}",
            'imshow_ms': f"{timings.get('imshow', 0):.2f}",
            'memory_mb': f'{memory_mb:.2f}'
        }
        self.log(data)


class GestureLogger(CSVLogger):
    """ジェスチャー検出ログを記録"""
    
    def __init__(self, filepath: str):
        headers = [
            'timestamp',
            'state',
            'detected',
            'hand_x',
            'hand_y',
            'confidence',
            'countdown_active',
            'frame_number'
        ]
        super().__init__(filepath, headers)
    
    def log_detection(self, state: str, detected: bool, 
                     hand_pos: Optional[tuple] = None,
                     confidence: float = 0.0,
                     countdown_active: bool = False,
                     frame_number: int = 0):
        """ジェスチャー検出結果を記録
        
        Args:
            state: 現在の状態
            detected: ジェスチャーが検出されたか
            hand_pos: 手の位置 (x, y) または None
            confidence: 検出信頼度（0.0-1.0）
            countdown_active: カウントダウン中か
            frame_number: フレーム番号
        """
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'state': state,
            'detected': 'True' if detected else 'False',
            'hand_x': f'{hand_pos[0]:.2f}' if hand_pos else '',
            'hand_y': f'{hand_pos[1]:.2f}' if hand_pos else '',
            'confidence': f'{confidence:.3f}',
            'countdown_active': 'True' if countdown_active else 'False',
            'frame_number': str(frame_number)
        }
        self.log(data)


class DistanceLogger(CSVLogger):
    """距離調整とロボット動作ログを記録"""
    
    def __init__(self, filepath: str):
        headers = [
            'timestamp',
            'edges_detected',
            'bbox_center_x',
            'bbox_width',
            'bbox_height',
            'frame_center',
            'error_px',
            'error_percent',
            'robot_action',
            'motor_speed',
            'frame_number'
        ]
        super().__init__(filepath, headers)
    
    def log_adjustment(self, edges: set, bbox: Optional[tuple],
                      frame_width: int, frame_center: float,
                      robot_action: str, motor_speed: float = 0.0,
                      frame_number: int = 0):
        """距離調整とロボット動作を記録
        
        Args:
            edges: 検出されたエッジのセット (e.g., {'TOP', 'LEFT'})
            bbox: バウンディングボックス (min_x, max_x, min_y, max_y) または None
            frame_width: フレームの幅
            frame_center: フレームの中心X座標
            robot_action: ロボットの動作 ('FORWARD', 'BACKWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP', 'NONE')
            motor_speed: モーター速度
            frame_number: フレーム番号
        """
        bbox_center_x = 0
        bbox_width = 0
        bbox_height = 0
        error_px = 0
        error_percent = 0
        
        if bbox:
            min_x, max_x, min_y, max_y = bbox
            bbox_center_x = (min_x + max_x) / 2
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
            error_px = abs(bbox_center_x - frame_center)
            error_percent = (error_px / frame_width) * 100
        
        edges_str = ','.join(sorted(edges)) if edges else 'NONE'
        
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'edges_detected': edges_str,
            'bbox_center_x': f'{bbox_center_x:.2f}' if bbox else '',
            'bbox_width': f'{bbox_width:.2f}' if bbox else '',
            'bbox_height': f'{bbox_height:.2f}' if bbox else '',
            'frame_center': f'{frame_center:.2f}',
            'error_px': f'{error_px:.2f}' if bbox else '',
            'error_percent': f'{error_percent:.2f}' if bbox else '',
            'robot_action': robot_action,
            'motor_speed': f'{motor_speed:.2f}',
            'frame_number': str(frame_number)
        }
        self.log(data)


class LoggerManager:
    """全てのロガーを管理するマネージャークラス"""
    
    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self.session_dir = self._create_session_dir()
        
        # ロガーの初期化
        self.performance_logger = None
        self.gesture_logger = None
        self.distance_logger = None
        
        if self.config.enable_performance:
            perf_path = os.path.join(self.session_dir, 'performance.csv')
            self.performance_logger = PerformanceLogger(perf_path)
            print(f"[Logger] Performance log: {perf_path}")
        
        if self.config.enable_gesture:
            gesture_path = os.path.join(self.session_dir, 'gesture.csv')
            self.gesture_logger = GestureLogger(gesture_path)
            print(f"[Logger] Gesture log: {gesture_path}")
        
        if self.config.enable_distance:
            distance_path = os.path.join(self.session_dir, 'distance.csv')
            self.distance_logger = DistanceLogger(distance_path)
            print(f"[Logger] Distance log: {distance_path}")
        
        self.start_time = datetime.now()
        print(f"[Logger] Session started: {self.session_dir}")
    
    def _create_session_dir(self) -> str:
        """セッション用のディレクトリを作成"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f"{self.config.session_prefix}_{timestamp}"
        session_path = os.path.join(self.config.base_dir, session_name)
        os.makedirs(session_path, exist_ok=True)
        return session_path
    
    def log_performance(self, fps: float, frame_time_ms: float, 
                       state: str, timings: Dict[str, float]):
        """パフォーマンスログを記録"""
        if self.performance_logger:
            self.performance_logger.log_frame(fps, frame_time_ms, state, timings)
    
    def log_gesture(self, state: str, detected: bool,
                   hand_pos: Optional[tuple] = None,
                   confidence: float = 0.0,
                   countdown_active: bool = False,
                   frame_number: int = 0):
        """ジェスチャーログを記録"""
        if self.gesture_logger:
            self.gesture_logger.log_detection(
                state, detected, hand_pos, confidence, 
                countdown_active, frame_number
            )
    
    def log_distance(self, edges: set, bbox: Optional[tuple],
                    frame_width: int, frame_center: float,
                    robot_action: str, motor_speed: float = 0.0,
                    frame_number: int = 0):
        """距離調整ログを記録"""
        if self.distance_logger:
            self.distance_logger.log_adjustment(
                edges, bbox, frame_width, frame_center,
                robot_action, motor_speed, frame_number
            )
    
    def close(self):
        """全てのロガーを閉じてサマリーを保存"""
        if self.performance_logger:
            self.performance_logger.close()
        if self.gesture_logger:
            self.gesture_logger.close()
        if self.distance_logger:
            self.distance_logger.close()
        
        # セッションサマリーを保存
        self._save_summary()
        print(f"[Logger] Session ended: {self.session_dir}")
    
    def _save_summary(self):
        """セッションのサマリーをJSONで保存"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        summary = {
            'session_dir': self.session_dir,
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': duration,
            'logs': {
                'performance': self.config.enable_performance,
                'gesture': self.config.enable_gesture,
                'distance': self.config.enable_distance
            }
        }
        
        summary_path = os.path.join(self.session_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
