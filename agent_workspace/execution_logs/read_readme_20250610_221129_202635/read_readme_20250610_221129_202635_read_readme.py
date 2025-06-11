
import os

readme_path = "/sandbox/datasets/README.md"

if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    print("README.md 内容:\n" + readme_content)
else:
    print(f"错误: 文件 {readme_path} 不存在。")
