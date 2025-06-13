
import os

dataset_path = "/sandbox/datasets/"

# 1. List all files and first-level subdirectories in /sandbox/datasets/
print(f"Content of {dataset_path}:")
try:
    with os.scandir(dataset_path) as entries:
        for entry in entries:
            print(f"- {entry.name} {"/" if entry.is_dir() else ""}")
except FileNotFoundError:
    print(f"Error: Directory {dataset_path} not found. Please ensure your datasets are mounted correctly.")
except Exception as e:
    print(f"An error occurred while scanning {dataset_path}: {e}")

# 2. Check for common descriptive files and read them
descriptive_files = [
    "README.md",
    "README.txt",
    "data_description.txt",
    "description.md",
    "schema.json",
    "dataset_info.txt"
]

print("\nChecking for descriptive files:")
for filename in descriptive_files:
    file_path = os.path.join(dataset_path, filename)
    if os.path.exists(file_path):
        print(f"- Found {filename}. Reading content:")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(1000)  # Read up to 1000 characters
                print(content)
                if len(content) == 1000:
                    print(f"... (truncated, file content might be longer)")
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
    else:
        print(f"- {filename} not found.")
