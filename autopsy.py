import torch
from torchvision.ops import deform_conv2d

print('[1] 正在 RTX 5060 Ti 上分配测试张量...')
try:
    x = torch.randn(2, 64, 32, 32).cuda()
    weight = torch.randn(64, 64, 3, 3).cuda()
    offset = torch.randn(2, 18, 32, 32).cuda()
    mask = torch.rand(2, 9, 32, 32).cuda()
    
    print('[2] 张量分配成功，准备冲击 C++ 底层 DCN 算子 (危险区)...')
    y = deform_conv2d(x, offset, weight, padding=(1, 1), mask=mask)
    
    print('[3] 算子存活！输出维度:', y.shape)
    print('=== 结论：算子没坏，是 YOLO 的问题 ===')
except Exception as e:
    print(f'[FATAL ERROR] {e}')
