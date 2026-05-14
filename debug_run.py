import os
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# 最基础的注入，不使用正则，手动硬挂载
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import C2f, SPPF
# 这里假设你的类定义已经在当前环境或你能手动粘贴的地方
# 为了绝对成功，我们先跑一个最简单的逻辑检查
if __name__ == '__main__':
    print('[CHECK] Starting Minimal Debug Mode...')
    # 1. 物理清理缓存
    if os.path.exists(r'D:\aconda\polyp_data\labels\train.cache'):
        os.remove(r'D:\aconda\polyp_data\labels\train.cache')
        print('[CHECK] Old cache cleaned.')

    # 2. 强制使用最基础的 Batch=4 探测极限
    model = YOLO(r'D:\medical\yolov8n-NormAImed_DCN.yaml', task='detect')
    model.train(
        data=r'D:\aconda\polyp_data.yaml',
        epochs=1,
        batch=4,           # 极低 Batch，排除内存问题
        workers=0,         # 绝对单线程
        amp=False,         # 关闭混合精度，排除算子兼容性
        cache=False,       # 不缓存到内存
        device=0,
        exist_ok=True
    )
