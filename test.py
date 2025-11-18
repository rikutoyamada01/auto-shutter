import os
import subprocess
import sys

VENV_DIR = ".venv"
TEST_COMMAND = "pytest" # または "python -m unittest discover" など

def run_tests_in_venv():
    """
    仮想環境を有効化し、テストを実行します。
    """
    # 仮想環境内のPythonインタープリタのパスを決定
    if sys.platform == "win32":
        python_executable = os.path.join(VENV_DIR, "Scripts", "python.exe")
        pip_executable = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else: # linux or darwin (macOS)
        python_executable = os.path.join(VENV_DIR, "bin", "python")
        pip_executable = os.path.join(VENV_DIR, "bin", "pip")

    # 仮想環境のPythonが存在するか確認
    if not os.path.exists(python_executable):
        print(f"エラー: 仮想環境のPythonインタープリタが見つかりません: {python_executable}")
        print("先に `python setup.py` を実行して仮想環境をセットアップしてください。")
        sys.exit(1)

    # pytestがインストールされているか確認し、なければインストール
    try:
        subprocess.run([python_executable, "-m", "pytest", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("pytestが見つかりません。仮想環境にインストールします...")
        try:
            subprocess.run([pip_executable, "install", "pytest"], check=True)
            print("pytestをインストールしました。")
        except subprocess.CalledProcessError as e: # Changed from subprocess.CalledProcessError to subprocess.CalledPythonError
            print(f"エラー: pytestのインストールに失敗しました。")
            print(f"詳細: {e.stderr.decode()}")
            sys.exit(1)
    
    print(f"仮想環境 ({VENV_DIR}) を使用してテストを実行します...")
    
    try:
        # 仮想環境のPythonを使って pytest を実行
        # テストファイルはまだ存在しないが、pytestは引数なしで実行するとカレントディレクトリ以下のテストを探索する
        subprocess.run([python_executable, "-m", "pytest"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"エラー: テストの実行中に問題が発生しました。")
        print(f"詳細: {e}")
        # pytestが失敗した場合でも、スクリプト自体は成功として終了させない
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"エラー: '{python_executable}' が見つかりません。")
        sys.exit(1)

if __name__ == "__main__":
    run_tests_in_venv()
