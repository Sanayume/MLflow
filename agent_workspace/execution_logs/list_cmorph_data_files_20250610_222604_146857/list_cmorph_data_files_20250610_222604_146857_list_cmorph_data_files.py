
import os

try:
    files = os.listdir('/sandbox/datasets/rainfalldata/CMORPHdata/')
    print(f"Files in /sandbox/datasets/rainfalldata/CMORPHdata/: {files}")
except FileNotFoundError:
    print("Error: The directory /sandbox/datasets/rainfalldata/CMORPHdata/ was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
