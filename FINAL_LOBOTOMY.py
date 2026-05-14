import os
import re
import sys
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as torch_utils
import inspect

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
    def __init__(self, *args):
        super().__init__()
        self.ca = None
        self.sa = None
    def forward(self, x):
        if self.ca is None:
            self.ca = ChannelAttention(x.shape[1]).to(x.device, x.dtype)
            self.sa = SpatialAttention().to(x.device, x.dtype)
        return self.sa(self.ca(x))

class DCNv3_Native(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=bias)
        self.offset_conv = nn.Conv2d(c1, 2 * k * k, k, s, p)
        self.mask_conv = nn.Conv2d(c1, k * k, k, s, p)
    def forward(self, x):
        x_c = x.contiguous()
        offset = self.offset_conv(x_c).contiguous()
        mask = torch.sigmoid(self.mask_conv(x_c)).contiguous()
        return deform_conv2d(x_c, offset, self.conv.weight, mask=mask, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)

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

class SE_Wrapper(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.module = None
    def forward(self, x):
        if self.module is None:
            c = x.shape[1]
            self.module = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, max(1, c // 16), 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, c // 16), c, 1, bias=False),
                nn.Sigmoid()
            ).to(x.device, x.dtype)
        return x * self.module(x)

class ECA_Wrapper(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.module = None
    def forward(self, x):
        if self.module is None:
            import math
            c = x.shape[1]
            k = int(abs((math.log(c, 2) + 1) / 2))
            k = k if k % 2 else k + 1
            self.module = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False).to(x.device, x.dtype)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.sigmoid = nn.Sigmoid()
        y = self.avg_pool(x)
        y = self.module(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class CA_Wrapper(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.module = None
    def forward(self, x):
        if self.module is None:
            c = x.shape[1]
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
            mip = max(8, c // 32)
            self.conv1 = nn.Conv2d(c, mip, kernel_size=1, stride=1, padding=0).to(x.device, x.dtype)
            self.bn1 = nn.BatchNorm2d(mip).to(x.device, x.dtype)
            self.act = nn.SiLU().to(x.device, x.dtype)
            self.conv_h = nn.Conv2d(mip, c, kernel_size=1, stride=1, padding=0).to(x.device, x.dtype)
            self.conv_w = nn.Conv2d(mip, c, kernel_size=1, stride=1, padding=0).to(x.device, x.dtype)
            self.sigmoid = nn.Sigmoid()
            self.module = True
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w

class EMA_Wrapper(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.module = None
    def forward(self, x):
        if self.module is None:
            c = x.shape[1]
            self.groups = c // 8 if c >= 8 else 1
            self.conv1x1 = nn.Conv2d(c // self.groups, c // self.groups, kernel_size=1).to(x.device, x.dtype)
            self.conv3x3 = nn.Conv2d(c // self.groups, c // self.groups, kernel_size=3, padding=1).to(x.device, x.dtype)
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
            self.sigmoid = nn.Sigmoid()
            self.module = True
        b, c, h, w = x.shape
        group_x = x.reshape(b * self.groups, c // self.groups, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x1 = self.sigmoid(x_h * x_w) * group_x
        x2 = self.conv3x3(group_x)
        return (x1 + x2).reshape(b, c, h, w)

class GAM_Wrapper(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.module = None
    def forward(self, x):
        if self.module is None:
            c = x.shape[1]
            self.channel_attention = nn.Sequential(
                nn.Conv2d(c, c // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 4, c, 1),
                nn.Sigmoid()
            ).to(x.device, x.dtype)
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(c, c // 4, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 4, c, 7, padding=3),
                nn.Sigmoid()
            ).to(x.device, x.dtype)
            self.module = True
        x_c = x * self.channel_attention(x)
        return x_c * self.spatial_attention(x_c)

# 【劫持 1】：注入算子
tasks.__dict__.update({'BiFPN_Concat': BiFPN_Concat, 'CBAM': CBAM, 'C2f_DCNv3': C2f_DCNv3, 'SE_Wrapper': SE_Wrapper, 'ECA_Wrapper': ECA_Wrapper, 'CA_Wrapper': CA_Wrapper, 'EMA_Wrapper': EMA_Wrapper, 'GAM_Wrapper': GAM_Wrapper})
source = inspect.getsource(tasks.parse_model)
source = re.sub(r'\bC2f\b', 'C2f, C2f_DCNv3', source)
exec(source, tasks.__dict__)

# 【劫持 2】：物理切除 YOLO 算力测算模块 (Lobotomy)
torch_utils.get_flops = lambda *args, **kwargs: 0.0
torch_utils.get_flops_with_params = lambda *args, **kwargs: (0.0, 0.0)

if __name__ == '__main__':
    print('[SYSTEM] Executing YOLO Lobotomy & Launching Training...')
    torch.cuda.empty_cache()
    
    model = YOLO('./yolov8n-NormAImed_DCN.yaml', task='detect')
    
    # 彻底封死 model.info 的执行路径，防止假张量撞爆 C++ 底层
    model.info = lambda *args, **kwargs: None
    
    model.train(
        data='./datasets/polyp_data.yaml', 
        epochs=100, 
        imgsz=640, 
        batch=32, 
        device=0, 
        project='./ablation_runs', 
        name='Final_DCN_100e_Lobotomy', 
        amp=False, 
        workers=0
    )