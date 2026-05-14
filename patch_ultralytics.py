import sys
import os

conv_path = r'D:\aconda\envs\yolo_env\Lib\site-packages\ultralytics\nn\modules\conv.py'
tasks_path = r'D:\aconda\envs\yolo_env\Lib\site-packages\ultralytics\nn\tasks.py'

def modify_conv():
    with open(conv_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'from torchvision.ops import deform_conv2d' not in content:
        content = content.replace('import torch.nn as nn', 'import torch.nn as nn\nfrom torchvision.ops import deform_conv2d')
    
    if 'C2f_DCNv3' not in content:
        # Add C2f_DCNv3 to __all__
        content = content.replace('"SpatialAttention",', '"SpatialAttention",\n    "C2f_DCNv3",')
        
        # Add classes at the end
        new_classes = """

class DCNv3_Native(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.stride = s
        self.padding = p
        self.dilation = d
        self.groups = g
        self.offset_mask_conv = nn.Conv2d(c1, 3 * g * k * k, kernel_size=k, stride=s, padding=self.padding, dilation=d, bias=True)
        self.regular_conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=self.padding, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        offset_mask = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = deform_conv2d(x, offset, self.regular_conv.weight, self.regular_conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask)
        return self.act(self.bn(x))

class Bottleneck_DCNv3(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DCNv3_Native(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_DCNv3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_DCNv3(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
"""
        content += new_classes
        
    with open(conv_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("conv.py modified successfully.")

def modify_tasks():
    with open(tasks_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'C2f_DCNv3' not in content:
        # Import
        content = content.replace('from ultralytics.nn.modules import (', 'from ultralytics.nn.modules import (\n    C2f_DCNv3,')
        
        # Add to base_modules
        content = content.replace('A2C2f,\n        }', 'A2C2f,\n            C2f_DCNv3,\n        }')
        
        # Add to repeat_modules
        content = content.replace('A2C2f,\n        }', 'A2C2f,\n            C2f_DCNv3,\n        }')
        
    with open(tasks_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("tasks.py modified successfully.")

try:
    modify_conv()
    modify_tasks()
except Exception as e:
    print(f"Error: {e}")
