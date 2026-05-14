import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据映射区 (请替换为 5.4 节真实数据)
# ==========================================
models = ['YOLOv8n', 'YOLOv8m', 'Norm-AImed (Ours)']
flops = [8.2, 79.1, 8.2]  # X轴: 算力 FLOPs (G) - 越小越好
map_val = [72.8, 66.5, 83.8]  # Y轴: 精度 mAP@0.5:0.95 (%) - 越大越好
fps = [297.0, 115.3, 225.7]  # 气泡大小: 全链路 FPS - 越大说明实时性越强

# ==========================================
# 2. 学术视觉美学配置
# ==========================================
# 核心视觉锚点：克莱因蓝
COLOR_OURS = '#002FA7'
# 对比组视觉降维：冷灰色
COLOR_BASE = '#A0A0A0'
colors = [COLOR_BASE, COLOR_BASE, COLOR_OURS]

# 气泡基础尺寸缩放系数 (根据实际画面比例微调)
bubble_sizes = [f * 6 for f in fps]

# 初始化高分辨率画布 (DPI=300 符合学术期刊付印标准)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# ==========================================
# 3. 核心图表绘制
# ==========================================
scatter = ax.scatter(flops, map_val, s=bubble_sizes, c=colors,
                     alpha=0.85, edgecolors='white', linewidth=1.5)

# 动态添加高对比度数据标签
for i, txt in enumerate(models):
    # 根据模型是否为核心模型，应用不同字重与颜色强化视觉层级
    font_weight = 'bold' if 'Ours' in txt else 'normal'

    ax.annotate(f"{txt}\n({fps[i]} FPS)",
                (flops[i], map_val[i]),
                xytext=(0, 20), textcoords='offset points',
                ha='center', va='bottom', fontsize=10,
                fontweight=font_weight, color=colors[i])

# ==========================================
# 4. 极简坐标系与轴线重构
# ==========================================
ax.set_xlabel('FLOPs (G) [Lower is Better]', fontsize=12, fontweight='bold', color='#333333')
ax.set_ylabel('mAP@0.5:0.95 (%) [Higher is Better]', fontsize=12, fontweight='bold', color='#333333')
ax.set_title('Accuracy-Compute Pareto Frontier (RTX 5060 Ti)', fontsize=14, fontweight='bold', pad=25)

# 物理剥离冗余的顶部与右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 强化数据包络底线
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_color('#333333')

# 仅保留 Y 轴水平虚线，提供横向精度对齐锚点，切断 X 轴网格防视觉干扰
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
# plt.savefig('pareto_frontier_bubble.pdf', format='pdf', bbox_inches='tight') # 取消注释以保存为矢量图
plt.show()