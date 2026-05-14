import os
import json
import random
import shutil
import cv2
from pathlib import Path

# --- 1. 路径配置 ---
root_dir = Path("F:/medical")
src_images = root_dir / "Kvasir-SEG/images"
json_file = root_dir / "Kvasir-SEG/kavsir_bboxes.json"
target_dir = root_dir / "yolo_dataset"

# --- 2. 建立标准文件夹 ---
for s in ['train', 'val']:
    (target_dir / 'images' / s).mkdir(parents=True, exist_ok=True)
    (target_dir / 'labels' / s).mkdir(parents=True, exist_ok=True)

# --- 3. 读取并划分数据 ---
with open(json_file, 'r') as f:
    data = json.load(f)

img_keys = list(data.keys())
random.shuffle(img_keys)
split_idx = int(len(img_keys) * 0.8)
train_keys = img_keys[:split_idx]
val_keys = img_keys[split_idx:]


# --- 4. 核心处理函数 (注意这里的缩进) ---
def process_data(keys, subset):
    print(f"正在处理 {subset} 集，共 {len(keys)} 张图片...")
    for key in keys:
        item = data[key]
        img_name = f"{key}.jpg"
        src_img_path = src_images / img_name

        if not src_img_path.exists():
            continue

        # 获取图片尺寸用于归一化
        img = cv2.imread(str(src_img_path))
        if img is None: continue
        h, w, _ = img.shape

        yolo_annotations = []
        raw_bboxes = item.get('bbox', [])

        # 遍历 JSON 中的字典列表
        for box_dict in raw_bboxes:
            xmin, ymin = box_dict['xmin'], box_dict['ymin']
            xmax, ymax = box_dict['xmax'], box_dict['ymax']

            # 计算 YOLO 归一化坐标
            bw = xmax - xmin
            bh = ymax - ymin
            x_center = (xmin + bw / 2) / w
            y_center = (ymin + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h

            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        # 物理分发文件
        shutil.copy(str(src_img_path), target_dir / 'images' / subset / img_name)
        with open(target_dir / 'labels' / subset / f"{key}.txt", 'w') as f_txt:
            f_txt.write("\n".join(yolo_annotations))


# --- 5. 执行 ---
if __name__ == "__main__":
    process_data(train_keys, 'train')
    process_data(val_keys, 'val')
    print(f"\n恭喜！数据已成功分发至: {target_dir}")