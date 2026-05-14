import os
import json
import cv2
from pathlib import Path

# ==================== 配置区 ====================
# 1. 你的 JSON 标注文件路径
JSON_PATH = r"F:\medical\Kvasir-SEG\kavsir_bboxes.json"

# 2. 你当前的训练集图片文件夹路径
TRAIN_IMAGES_DIR = r"F:\medical\yolo_dataset\images\train"

# 3. 你当前的验证集图片文件夹路径
VAL_IMAGES_DIR = r"F:\medical\yolo_dataset\images\val"


# ===============================================

def process_labels(image_dir, json_data):
    image_dir = Path(image_dir)
    # 自动根据 images 的位置推导出 labels 的位置
    # 逻辑：在 images 的同级创建 labels，并在 labels 下创建与 images 子目录同名的文件夹
    label_dir = image_dir.parent.parent / "labels" / image_dir.name
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在为 {image_dir.name} 生成标签，目标路径: {label_dir}")

    # 获取该文件夹下所有图片的文件名（不含后缀）
    image_files = [f.stem for f in image_dir.glob("*.jpg")]
    count = 0

    for key in image_files:
        if key in json_data:
            item = json_data[key]
            img_path = image_dir / f"{key}.jpg"

            # 读取图片以获取其实际宽高（用于归一化）
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w, _ = img.shape

            yolo_annotations = []
            raw_bboxes = item.get('bbox', [])

            for box_dict in raw_bboxes:
                # 提取 JSON 中的像素坐标
                xmin, ymin = box_dict['xmin'], box_dict['ymin']
                xmax, ymax = box_dict['xmax'], box_dict['ymax']

                # 计算 YOLO 要求的中心点归一化坐标
                bw = xmax - xmin
                bh = ymax - ymin
                x_center = (xmin + bw / 2) / w
                y_center = (ymin + bh / 2) / h
                norm_w = bw / w
                norm_h = bh / h

                # 格式：类别(0) x_center y_center width height
                yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            # 写入 TXT 文件
            with open(label_dir / f"{key}.txt", 'w') as f:
                f.write("\n".join(yolo_annotations))
            count += 1

    print(f"完成！成功生成 {count} 个标签文件。")


if __name__ == "__main__":
    # 加载 JSON
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # 分别处理两个文件夹
    process_labels(TRAIN_IMAGES_DIR, data)
    process_labels(VAL_IMAGES_DIR, data)