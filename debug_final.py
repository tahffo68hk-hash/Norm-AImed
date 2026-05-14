import os
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    print('[SYSTEM] Starting Minimal Debug Mode (Batch=16)...')
    
    # 强制清理可能损坏的缓存索引
    cache_path = r'D:\aconda\polyp_data\labels\train.cache'
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print('[SYSTEM] Corrupted cache cleared.')

    # 极简启动：排除所有干扰项
    try:
        model = YOLO(r'D:\medical\yolov8n-NormAImed_DCN.yaml', task='detect')
        model.train(
            data=r'D:\aconda\polyp_data.yaml',
            epochs=100,
            batch=16,
            workers=0,
            device=0,
            amp=True,
            exist_ok=True
        )
    except Exception as e:
        print(f'\n[ERROR] {e}')
