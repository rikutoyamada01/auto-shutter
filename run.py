import os
import subprocess
import sys

VENV_DIR = ".venv"
MAIN_APP = "src/main.py"

def run_in_venv():
    """
    仮想環境を有効化し、main.py を実行します。
    """
    # 仮想環境内のPythonインタープリタのパスを決定
    if sys.platform == "win32":
        python_executable = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else: # linux or darwin (macOS)
        python_executable = os.path.join(VENV_DIR, "bin", "python")

    # 仮想環境のPythonが存在するか確認
    if not os.path.exists(python_executable):
        print(f"エラー: 仮想環境のPythonインタープリタが見つかりません: {python_executable}")
        print("先に `python setup.py` を実行して仮想環境をセットアップしてください。")
        sys.exit(1)

    # main.py が存在するか確認
    if not os.path.exists(MAIN_APP):
        print(f"エラー: アプリケーションのメインファイルが見つかりません: {MAIN_APP}")
        print("srcの中に`main.py` が存在することを確認してください。")
        sys.exit(1)

    print(f"仮想環境 ({VENV_DIR}) を使用して {MAIN_APP} を実行します...")
    
    try:
        # 仮想環境のPythonを使って main.py を実行
        subprocess.run([python_executable, MAIN_APP], check=True)
    except subprocess.CalledProcessError as e:
        print(f"エラー: アプリケーションの実行中に問題が発生しました。")
        print(f"詳細: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"エラー: '{python_executable}' または '{MAIN_APP}' が見つかりません。")
        sys.exit(1)

if __name__ == "__main__":
    run_in_venv()
