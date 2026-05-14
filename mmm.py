from pathlib import Path

# 1. 定义根目录（你的移动硬盘路径）
root = Path("F:/medical/yolo_dataset")

# 2. 定义需要创建的所有子路径
folders = [
    root / "images/train",
    root / "images/val",
    root / "labels/train",
    root / "labels/val"
]

# 3. 循环创建
for folder in folders:
    # parents=True: 如果 medical 文件夹不存在，它会自动帮你建好
    # exist_ok=True: 如果文件夹已经存在了，它不会报错报错，直接跳过
    folder.mkdir(parents=True, exist_ok=True)
    print(f"已就绪: {folder}")

print("\n>>> 所有 YOLO 标准文件夹已建立完毕。")