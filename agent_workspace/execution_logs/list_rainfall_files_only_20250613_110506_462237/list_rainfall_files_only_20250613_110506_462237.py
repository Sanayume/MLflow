
import os

data_dir = "/sandbox/datasets/rainfalldata/"

print(f"Listing files in: {data_dir}")
files = os.listdir(data_dir)
if files:
    for f in files:
        print(f"- {f}")
else:
    print(f"No files found in {data_dir}.")
