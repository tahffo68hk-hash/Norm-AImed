import torch
import torch.nn as nn
import sys
import os
from torchvision.ops import deform_conv2d
from pathlib import Path

# [ignoring loop detection]

# =========================================================
# 1. 核心类定义 (Full Definitions)
# =========================================================

class BiFPN_Concat(nn.Module):
    """自适应特征融合的 BiFPN Concat 模块"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        return torch.cat(x, self.d)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    """用于医疗影像精准定位的 CBAM 注意力机制"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x))

class DCNv3_Native(nn.Module):
    """基于 torchvision 原生算子的 DCNv3 适配器"""
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=dilation, bias=bias)
        self.offset_conv = nn.Conv2d(c1, 2 * k * k, k, s, p)
        self.mask_conv = nn.Conv2d(c1, k * k, k, s, p)
    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return deform_conv2d(x, offset, self.conv.weight, mask=mask, 
                             stride=self.conv.stride, padding=self.conv.padding, 
                             dilation=self.conv.dilation)

class Bottleneck_DCNv3(nn.Module):
    """带有 DCNv3 算子的瓶颈结构"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1) 
        self.cv2 = DCNv3_Native(c_, c2, k[1], 1, 1, g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_DCNv3(nn.Module):
    """核心自定义 Backbone/Neck 模块"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_DCNv3(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# =========================================================
# 2. 深度猴子补丁 (Deep Monkey Patching)
# =========================================================

def apply_nuclear_patch():
    print("[SYSTEM] Starting Deep Monkey Patching...")
    
    custom_classes = {
        'BiFPN_Concat': BiFPN_Concat,
        'CBAM': CBAM,
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention,
        'C2f_DCNv3': C2f_DCNv3,
        'Bottleneck_DCNv3': Bottleneck_DCNv3,
        'DCNv3_Native': DCNv3_Native
    }

    target_module_paths = [
        'ultralytics.nn.tasks',
        'ultralytics.nn.modules.conv',
        'ultralytics.nn.modules.block',
        'ultralytics.nn.modules'
    ]

    import importlib
    for path in target_module_paths:
        try:
            mod = importlib.import_module(path)
            for name, cls in custom_classes.items():
                setattr(mod, name, cls)
                if path in sys.modules:
                    setattr(sys.modules[path], name, cls)
            print(f"[SUCCESS] Injected custom classes into {path}")
        except Exception as e:
            print(f"[ERROR] Failed to patch {path}: {e}")

# =========================================================
# 3. 导出执行逻辑 (Execution)
# =========================================================

if __name__ == "__main__":
    apply_nuclear_patch()
    
    from ultralytics import YOLO

    # 优先使用命令行指定的路径，其次查找当前目录下的 best.pt
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
    else:
        weights_path = "best.pt"
        
    if not os.path.exists(weights_path):
        alt_path = "runs/detect/train/weights/best.pt"
        if os.path.exists(alt_path):
            weights_path = alt_path
        else:
            print(f"[CRITICAL] Weights file {weights_path} NOT found!")
            sys.exit(1)

    print(f"[INFO] Initializing model with deep-patched environment: {weights_path}")
    
    try:
        model = YOLO(weights_path)
    except AttributeError as e:
        print(f"[FATAL ERROR] Still missing attributes during loading: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL ERROR] Unknown error during model initialization: {e}")
        sys.exit(1)

    print("[INFO] Model loaded successfully. Starting TensorRT Export...")
    try:
        model.export(
            format='engine', 
            half=True, 
            device=0, 
            workspace=8, 
            simplify=True,
            opset=16
        )
        print("\n" + "="*50)
        print("[SUCCESS] TensorRT Engine has been forged successfully!")
        print("="*50)
    except Exception as e:
        print(f"\n[EXPORT FAILED] TensorRT compilation crashed: {e}")
