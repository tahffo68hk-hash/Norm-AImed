import os
import sys
import time
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

class BiFPN_Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x): return torch.cat(x, self.d)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, channels // reduction, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(channels // reduction, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x): return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

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
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x): return self.sa(self.ca(x))

class DCNv3_Native(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=bias)
        self.offset_conv = nn.Conv2d(c1, 2 * k * k, k, s, p)
        self.mask_conv = nn.Conv2d(c1, k * k, k, s, p)
    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return deform_conv2d(x, offset, self.conv.weight, mask=mask, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)

class Bottleneck_DCNv3(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = DCNv3_Native(c_, c2, k[1], 1, 1, g)
        self.add = shortcut and c1 == c2
    def forward(self, x): return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_DCNv3(nn.Module):
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

print('[SYSTEM] Executing direct global injection...')
tasks.__dict__.update({'BiFPN_Concat': BiFPN_Concat, 'CBAM': CBAM, 'C2f_DCNv3': C2f_DCNv3})
setattr(tasks, 'BiFPN_Concat', BiFPN_Concat)
setattr(tasks, 'CBAM', CBAM)
setattr(tasks, 'C2f_DCNv3', C2f_DCNv3)

try:
    print('[SYSTEM] Initiating Final DCNv3 Training...')
    model = YOLO(r'D:\medical\yolov8n-NormAImed_DCN.yaml', task='detect')
    results = model.train(data=r'D:\aconda\polyp_data.yaml', epochs=50, imgsz=640, batch=16, device=0, project='D:/medical/ablation_runs', name='Final_DCN', amp=True)
    print('\n[SUCCESS] Final DCNv3 Training Completed.')
except Exception as e:
    print(f'\n[FATAL ERROR] {e}')
