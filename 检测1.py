import json
from pathlib import Path

json_file = Path("F:/medical/Kvasir-SEG/kavsir_bboxes.json")
with open(json_file, 'r') as f:
    data = json.load(f)

# 打印前3个样本的 bbox 结构
for i, (key, value) in enumerate(data.items()):
    if i < 3:
        print(f"文件名: {key}")
        print(f"bbox 的原始值: {value['bbox']}")
        print(f"bbox 的类型: {type(value['bbox'])}")
        print("-" * 20)
    else:
        break