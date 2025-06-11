
import os

def list_files_in_directory(path):
    files = []
    dirs = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            files.append(item)
        elif os.path.isdir(item_path):
            dirs.append(item)
    return files, dirs

datasets_path = '/sandbox/datasets/'
print(f"Contents of {datasets_path}:")
try:
    files, dirs = list_files_in_directory(datasets_path)
    if files:
        print("Files:")
        for f in files:
            print(f"- {f}")
    else:
        print("No files found.")
    if dirs:
        print("Directories:")
        for d in dirs:
            print(f"- {d}/")
    else:
        print("No subdirectories found.")
except FileNotFoundError:
    print(f"Directory not found: {datasets_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# Also check for common descriptive files
descriptive_files = ['README.md', 'README.txt', 'data_description.txt', 'description.md', 'schema.json']
found_descriptive_files = []
print("\nChecking for common descriptive files:")
for df_name in descriptive_files:
    df_path = os.path.join(datasets_path, df_name)
    if os.path.exists(df_path) and os.path.isfile(df_path):
        found_descriptive_files.append(df_path)
        print(f"- Found: {df_path}")

if found_descriptive_files:
    print("\nReading content of found descriptive files (first 1000 characters):")
    for df_path in found_descriptive_files:
        try:
            with open(df_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)
                print(f"\n--- Content of {df_path} ---")
                print(content)
                if len(content) == 1000:
                    print("... (truncated)")
                print("--- End Content ---")
        except Exception as e:
            print(f"Could not read {df_path}: {e}")
else:
    print("No common descriptive files found.")
    