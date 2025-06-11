
import os

datasets_path = '/sandbox/datasets/'

print(f"正在列出 {datasets_path} 下的文件和目录...")

# 列出所有文件和一级子目录
items = os.listdir(datasets_path)
print("文件和目录列表:")
for item in items:
    full_path = os.path.join(datasets_path, item)
    if os.path.isfile(full_path):
        print(f"- 文件: {item}")
    elif os.path.isdir(full_path):
        print(f"- 目录: {item}/")

# 检查常见的描述性文件
description_files = [
    'README.md',
    'README.txt',
    'data_description.txt',
    'description.md',
    'schema.json',
    'dataset_info.txt'
]

print("\n正在检查描述性文件...")
for desc_file in description_files:
    desc_file_path = os.path.join(datasets_path, desc_file)
    if os.path.exists(desc_file_path) and os.path.isfile(desc_file_path):
        print(f"找到了描述性文件: {desc_file}")
        try:
            with open(desc_file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000) # 读取前1000个字符
                print(f"--- 内容 (前1000字符) ---\n{content}\n-----------------------")
        except Exception as e:
            print(f"--- 无法读取文件 {desc_file}: {e} ---")


