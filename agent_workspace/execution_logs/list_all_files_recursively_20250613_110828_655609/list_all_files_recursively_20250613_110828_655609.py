
import os

def list_all_files_recursively(start_path):
    all_files = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

datasets_path = '/sandbox/datasets/'
files_in_datasets = list_all_files_recursively(datasets_path)

if files_in_datasets:
    print("在 /sandbox/datasets/ 目录下递归找到的所有文件:")
    for f in files_in_datasets:
        print(f)
else:
    print("在 /sandbox/datasets/ 目录下未找到任何文件。")
