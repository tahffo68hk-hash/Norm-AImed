import os
import torch
import FINAL_LOBOTOMY  # 强行注入内存
from ultralytics import YOLO
import ultralytics.utils.torch_utils as torch_utils

torch_utils.get_flops = lambda *args, **kwargs: 0.0
torch_utils.get_flops_with_params = lambda *args, **kwargs: (0.0, 0.0)

# 强制写死映射关系，拒绝 OS 动态生成，避免路径幽灵
variants = [
    {"name": "Exp_SE", "yaml": "./yolov8n-NormAImed_DCN_SE.yaml"},
    {"name": "Exp_ECA", "yaml": "./yolov8n-NormAImed_DCN_ECA.yaml"},
    {"name": "Exp_CA", "yaml": "./yolov8n-NormAImed_DCN_CA.yaml"},
    {"name": "Exp_EMA", "yaml": "./yolov8n-NormAImed_DCN_EMA.yaml"},
    {"name": "Exp_GAM", "yaml": "./yolov8n-NormAImed_DCN_GAM.yaml"}
]

def run_ablation():
    print('[SYSTEM] Launching Clean Ablation Batch Process...')
    torch.cuda.empty_cache()

    for exp in variants:
        name = exp["name"]
        yaml_path = exp["yaml"]
        
        last_pt_path = os.path.join('./ablation_runs', name, 'weights', 'last.pt')
        
        try:
            if os.path.exists(last_pt_path):
                print(f"[SYSTEM] 发现 {name} 的有效存档！执行断点续传...")
                model = YOLO(last_pt_path, task='detect')
                model.info = lambda *args, **kwargs: None 
                model.train(resume=True)
            else:
                print(f"[SYSTEM] 启动 {name} 全新纯净训练...")
                model = YOLO(yaml_path, task='detect')
                model.info = lambda *args, **kwargs: None
                model.train(
                    data='./datasets/polyp_data.yaml',
                    epochs=100,
                    batch=32,
                    imgsz=640,
                    device=0,
                    amp=False,
                    workers=0,
                    project='./ablation_runs',
                    name=name
                )
            print(f"--- Completed training for {name} ---")
        except Exception as e:
            print(f"[ERROR] Failed to train {name}. Error: {e}")
            continue

if __name__ == '__main__':
    run_ablation()