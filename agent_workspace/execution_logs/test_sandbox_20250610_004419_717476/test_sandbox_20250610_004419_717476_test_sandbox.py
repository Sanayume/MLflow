
import os

output_dir = "/sandbox/outputs/"
output_file_path = os.path.join(output_dir, "test_output.txt")

print("Hello from the sandbox!")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 写入文件
with open(output_file_path, "w") as f:
    f.write("This is a test output from the sandbox.")

print(f"File written to: {output_file_path}")

# 读取文件并打印内容
with open(output_file_path, "r") as f:
    content = f.read()
    print(f"Content of {output_file_path}:")
    print(content)
