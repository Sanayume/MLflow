
import os

def list_files_recursively(startpath):
    print(f"Listing all files in {startpath} recursively:")
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

list_files_recursively("/sandbox/datasets/")
