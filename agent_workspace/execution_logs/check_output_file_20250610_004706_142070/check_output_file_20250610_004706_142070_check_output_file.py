
import os

output_dir = "/sandbox/outputs/"
file_name = "test_output.txt"
file_path = os.path.join(output_dir, file_name)

print(f"尝试检查文件: {file_path}")

if os.path.exists(file_path):
    print(f"文件 '{file_name}' 存在。")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        print(f"文件 '{file_name}' 的内容如下:\n---\n{content}\n---")
    except Exception as e:
        print(f"读取文件 '{file_name}' 时发生错误: {e}")
else:
    print(f"文件 '{file_name}' 不存在于 {output_dir}。")

print("列出 /sandbox/outputs/ 目录下的所有文件和文件夹:")
try:
    items = os.listdir(output_dir)
    if items:
        for item in items:
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path):
                print(f"  文件: {item}")
            elif os.path.isdir(item_path):
                print(f"  目录: {item}")
    else:
        print("  目录为空。")
except FileNotFoundError:
    print(f"  目录 {output_dir} 不存在。")
except Exception as e:
    print(f"  列出目录时发生错误: {e}")
