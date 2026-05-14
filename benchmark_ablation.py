import os
import sys
import csv
import time
import torch
import torch.nn as nn
import torchvision.ops as ops
from pathlib import Path
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.conv as conv_mod

# =====================================================================
# 1. 环境强制检查 (RTX 5060 Ti 性能释放准备)
# =====================================================================
assert torch.cuda.is_available(), "❌ 错误: 未检测到 CUDA 环境，请检查 NVIDIA 驱动！"
torch.backends.cudnn.benchmark = True # 开启 CUDNN 算子自动寻优

# =====================================================================
# 2. 核心模块定义 (DCNv3 & Bottleneck)
# =====================================================================

class DCNv3_Native(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None: p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.stride, self.padding, self.dilation, self.groups = s, p, d, g
        self.offset_mask_conv = nn.Conv2d(c1, 3 * g * k * k, k, s, p, d, bias=True)
        self.regular_conv = nn.Conv2d(c1, c2, k, s, p, d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = ops.deform_conv2d(x, offset, self.regular_conv.weight, self.regular_conv.bias,
                              self.stride, self.padding, self.dilation, mask)
        return self.act(self.bn(x))

class Bottleneck_DCNv3(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = conv_mod.Conv(c1, c_, k[0], 1)
        self.cv2 = DCNv3_Native(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_DCNv3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c1, c2, n = int(c1), int(c2), int(n)
        self.c = int(c2 * e)
        self.cv1 = conv_mod.Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = conv_mod.Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_DCNv3(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# =====================================================================
# 3. 动态劫持 (Monkey Patching)
# =====================================================================
setattr(tasks, 'C2f_DCNv3', C2f_DCNv3)
original_parse_model = tasks.parse_model

def patched_parse_model(d, ch, verbose=True):
    tasks.C2f_DCNv3 = C2f_DCNv3
    return original_parse_model(d, ch, verbose=False)

tasks.parse_model = patched_parse_model

# =====================================================================
# 4. 优化后的消融实验流程
# =====================================================================

def benchmark_model(config_path, data_path, name):
    print(f"\n🚀 测试模型: {name}")
    try:
        model = YOLO(config_path, task='detect')
        
        # --- 4.1 训练阶段 (优化 IO 线程) ---
        # 为演示快速运行，此处设为 5 epochs，用户可手动改为 50
        results = model.train(
            data=data_path,
            epochs=5, 
            imgsz=640,
            batch=16,
            device=0,
            workers=8, 
            project="./ablation_runs",
            name=f"Ablation_{name.replace(' ', '_')}",
            exist_ok=True,
            verbose=False,
            amp=True 
        )
        
        # --- 4.2 推理性能测量 (半精度 FP16 + CUDA 同步) ---
        print(f"[*] 正在执行 FP16 推理基准测试...")
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
        map50 = results.results_dict.get('metrics/mAP50(B)', 0.0)
        
        model.to('cuda').half()
        model.model.eval()
        
        dummy_input = torch.randn(1, 3, 640, 640).to('cuda').half()
        
        with torch.no_grad():
            for _ in range(50): _ = model.model(dummy_input)
        
        torch.cuda.synchronize()
        start_time = time.time()
        iters = 500
        with torch.no_grad():
            for _ in range(iters):
                _ = model.model(dummy_input)
        torch.cuda.synchronize()
        
        end_time = time.time()
        latency = ((end_time - start_time) / iters) * 1000
        fps = 1000 / latency
        
        print(f"📊 结果: mAP@0.5={map50:.4f} | FPS={fps:.1f} | Latency={latency:.2f}ms")
        
        return {
            "Model": name,
            "mAP@0.5": f"{map50:.4f}",
            "Params(M)": f"{params:.2f}",
            "Latency(ms)": f"{latency:.2f}",
            "FPS": f"{fps:.1f}"
        }
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return {"Model": name, "Error": str(e)}

def main():
    data_yaml = "./datasets/polyp_data.yaml"
    experiments = [
        ("yolov8n.yaml", "Baseline (YOLOv8n)"),
        ("./yolov8n-NormAImed.yaml", "Norm-AImed (BiFPN+CBAM)"),
        ("./yolov8n-NormAImed_DCN.yaml", "Final (Norm-AImed+DCNv3)")
    ]
    
    results_list = []
    for config, name in experiments:
        res = benchmark_model(config, data_yaml, name)
        results_list.append(res)
        
    output_csv = "./ablation_results.csv"
    keys = ["Model", "mAP@0.5", "Params(M)", "Latency(ms)", "FPS"]
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results_list)
        
    print(f"\n✅ 结果已保存至: {output_csv}")

if __name__ == "__main__":
    main()
