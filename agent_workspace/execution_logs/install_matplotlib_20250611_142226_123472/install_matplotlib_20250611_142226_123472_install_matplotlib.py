
import subprocess
import sys

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    print("matplotlib installed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error installing matplotlib: {e}")
    print(e.stdout)
    print(e.stderr)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
