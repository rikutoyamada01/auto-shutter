"""
ロギングシステムの動作確認テスト
"""
import os
import sys
import time

# src ディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from logger import LoggerManager, LogConfig


def test_logger():
    """ロガーの基本動作をテスト"""
    print("=== ロガーテスト開始 ===")
    
    # ロガーの初期化
    config = LogConfig()
    logger = LoggerManager(config)
    
    # パフォーマンスログのテスト
    print("\n[1] パフォーマンスログをテスト中...")
    for i in range(5):
        logger.log_performance(
            fps=30.0 - i,
            frame_time_ms=33.3 + i,
            state="READY",
            timings={
                'cap_read': 5.2,
                'process_state': 12.5,
                'draw_ui': 3.1,
                'imshow': 2.8
            }
        )
        time.sleep(0.1)
    print("   ✓ パフォーマンスログ記録完了")
    
    # ジェスチャーログのテスト
    print("\n[2] ジェスチャーログをテスト中...")
    for i in range(5):
        detected = i % 2 == 0
        logger.log_gesture(
            state="TAKE_PICTURE",
            detected=detected,
            hand_pos=(320.5 + i * 10, 240.3 - i * 5) if detected else None,
            confidence=0.95 - i * 0.05,
            countdown_active=i > 2,
            frame_number=100 + i
        )
        time.sleep(0.1)
    print("   ✓ ジェスチャーログ記録完了")
    
    # 距離調整ログのテスト
    print("\n[3] 距離調整ログをテスト中...")
    test_actions = ["FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
    test_edges = [set(), {"FAR"}, {"TOP"}, {"LEFT", "RIGHT"}, {"TOP", "LEFT"}]
    
    for i in range(5):
        bbox = (100 + i * 20, 500 - i * 20, 50, 400) if i % 2 == 0 else None
        logger.log_distance(
            edges=test_edges[i],
            bbox=bbox,
            frame_width=640,
            frame_center=320.0,
            robot_action=test_actions[i],
            motor_speed=0.3 if i < 3 else 0.0,
            frame_number=200 + i
        )
        time.sleep(0.1)
    print("   ✓ 距離調整ログ記録完了")
    
    # ロガーを閉じる
    print("\n[4] ロガーを閉じています...")
    logger.close()
    print("   ✓ ロガー終了")
    
    # 生成されたファイルを確認
    print(f"\n=== テスト完了 ===")
    print(f"\nログファイルの場所: {logger.session_dir}")
    print("\n生成されたファイル:")
    
    for root, dirs, files in os.walk(logger.session_dir):
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            print(f"  - {file} ({size} bytes)")
    
    print("\n✓ 全てのテストが正常に完了しました！")
    print(f"\nログファイルを確認してください: {logger.session_dir}")
    return logger.session_dir


if __name__ == "__main__":
    session_dir = test_logger()
    
    # ログディレクトリを開く（Windowsの場合）
    print(f"\nログディレクトリを開きますか？")
    print(f"ディレクトリ: {session_dir}")
