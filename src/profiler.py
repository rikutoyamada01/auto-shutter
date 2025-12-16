import time
from contextlib import contextmanager

class ProfileLogger:
    """
    指定されたブロックの実行時間を計測して出力するクラス
    使用例:
    with ProfileLogger("my_heavy_process"):
        # heavy process
        pass
    """
    def __init__(self, debug: bool = True):
        self.debug = debug

    @contextmanager
    def measure(self, label: str):
        if not self.debug:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            # 10ms以上かかった処理を目立たせるなどしても良いが、まずは全て出す
            print(f"[PROFILE] {label}: {elapsed:.4f} sec")

# シングルトンとしてインスタンス化（必要に応じてimportして使う）
profiler = ProfileLogger(debug=True)
