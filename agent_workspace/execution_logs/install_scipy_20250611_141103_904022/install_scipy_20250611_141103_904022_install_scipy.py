
import subprocess
import sys

try:
    # 尝试安装scipy
    process = subprocess.run([sys.executable, "-m", "pip", "install", "scipy"], capture_output=True, text=True)
    print("pip install scipy stdout:")
    print(process.stdout)
    print("pip install scipy stderr:")
    print(process.stderr)

    if process.returncode == 0:
        print("scipy installed successfully.")
    else:
        print(f"scipy installation failed with exit code {process.returncode}.")

except Exception as e:
    print(f"An error occurred during scipy installation: {e}")
