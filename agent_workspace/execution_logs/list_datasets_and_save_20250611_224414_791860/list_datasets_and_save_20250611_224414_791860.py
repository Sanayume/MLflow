
import os
import json

datasets_path = '/sandbox/datasets/'
items = os.listdir(datasets_path)
files = [item for item in items if os.path.isfile(os.path.join(datasets_path, item))]
dirs = [item for item in items if os.path.isdir(os.path.join(datasets_path, item))]

result_data = {
    "dataset_root_path": datasets_path,
    "total_items_found": len(items),
    "files_found": files,
    "directories_found": dirs,
    "description": "这是对 /sandbox/datasets/ 目录内容的初步扫描结果，用于测试数据库保存功能。"
}

print(json.dumps(result_data))
