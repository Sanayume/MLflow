
import os

datasets_path = '/sandbox/datasets/'

print(f"Listing contents of: {datasets_path}")
if not os.path.exists(datasets_path):
    print(f"Error: Directory {datasets_path} does not exist in the sandbox.")
else:
    for root, dirs, files in os.walk(datasets_path):
        level = root.replace(datasets_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')
