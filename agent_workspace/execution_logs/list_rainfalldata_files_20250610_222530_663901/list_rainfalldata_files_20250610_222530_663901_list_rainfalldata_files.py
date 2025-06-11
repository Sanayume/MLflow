
import os

try:
    files = os.listdir('/sandbox/datasets/rainfalldata/')
    print(f"Files in /sandbox/datasets/rainfalldata/: {files}")
except FileNotFoundError:
    print("Error: The directory /sandbox/datasets/rainfalldata/ was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
