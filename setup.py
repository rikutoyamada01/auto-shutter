import os
import sys
import subprocess
import json

# --- Configuration ---
VENV_DIR = ".venv"
SETTINGS_DIR = ".vscode"
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "settings.json")
REQUIREMENTS_FILE = "requirements.txt"
# ---------------------

def run_command(command, error_message):
    """Runs a shell command and exits if it fails."""
    try:
        # We use shell=True here for simplicity, especially on Windows
        # In a production environment, you might pass the command as a list.
        subprocess.run(command, check=True, shell=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {error_message}")
        print(f"Details: {e.stderr.decode()}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Command not found. Is '{command[0]}' installed and in your PATH?")
        sys.exit(1)

def main():
    print("--- Starting Project Setup ---")

    # 1. Create virtual environment if it doesn't exist
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment at '{VENV_DIR}'...")
        # sys.executable is the path to the current Python (e.g., python3 or python.exe)
        run_command(
            f"{sys.executable} -m venv {VENV_DIR}",
            "Failed to create virtual environment."
        )
    else:
        print(f"Virtual environment '{VENV_DIR}' already exists. Skipping.")

    # 2. Determine cross-platform paths for the venv
    if os.name == 'nt':  # Windows
        pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe")
        python_exe_setting = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:  # macOS / Linux
        pip_exe = os.path.join(VENV_DIR, "bin", "pip")
        python_exe_setting = os.path.join(VENV_DIR, "bin", "python")

    # 3. Install requirements using the venv's pip
    print(f"Installing dependencies from '{REQUIREMENTS_FILE}'...")
    run_command(
        f"{pip_exe} install -r {REQUIREMENTS_FILE}",
        "Failed to install requirements."
    )

    # 4. Create .vscode/settings.json to configure the editor
    print(f"Configuring VSCode settings at '{SETTINGS_FILE}'...")
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    
    # VSCode settings to auto-select the interpreter and enable type checking
    settings_data = {
        "python.defaultInterpreterPath": python_exe_setting,
        "python.analysis.typeCheckingMode": "basic"
    }
    
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings_data, f, indent=4)
    except IOError as e:
        print(f"Error: Failed to write settings file at '{SETTINGS_FILE}'.")
        print(f"Details: {e}")
        sys.exit(1)

    print("\n--- Setup Complete! ---")
    print("\nNext steps:")
    print("1. **Reload your VSCode window** (Cmd+Shift+P or Ctrl+Shift+P and type 'Developer: Reload Window').")
    print("2. VSCode should automatically select the '.venv' interpreter.")


if __name__ == "__main__":
    main()
