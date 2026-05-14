# Norm-AImed: A Real-time Endoscopic Polyp Detection System

[English](#norm-aimed-a-real-time-endoscopic-polyp-detection-system) | [中文](#norm-aimed-实时内窥镜息肉检测系统)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-green.svg)](https://github.com/ultralytics/ultralytics)

## Abstract
**Norm-AImed** is a high-performance, industrial-grade real-time polyp detection system designed for endoscopic procedures. Built upon an optimized **YOLOv8n** backbone, the system integrates state-of-the-art architectural enhancements to achieve superior accuracy and extreme inference speeds.

Key optimizations include:
- **WIoU v3**: Dynamic Non-monotonic Focusing for robust bounding box regression.
- **BiFPN_Concat**: Efficient multi-scale feature fusion for small lesion detection.
- **EMA Attention**: Cross-spatial Coordinate Attention to enhance feature localization.
- **DCNv3**: Deformable Convolutions to adapt to the irregular morphology of polyps.

The system achieves **225+ FPS** on an NVIDIA RTX 5060 Ti, with a **mAP@0.5 of 97.96%** on standard colorectal datasets.

---

## Core Features
- 🚀 **Extreme Inference Speed**: >225 FPS on consumer-grade GPUs, enabling lag-free real-time diagnostic assistance.
- 🎯 **Advanced Attention (EMA)**: Outperforms standard mechanisms (CBAM, SE, ECA) in medical imaging contexts by utilizing cross-spatial coordinate information.
- 🧬 **Morphological Adaptability**: DCNv3 integration allows the model to "reshape" its receptive field to match diverse polyp geometries.
- 🛡️ **OOM-Free Training**: Implements an asynchronous I/O queue and memory-optimized operators to ensure stable training even on limited VRAM.

---

## Performance Matrix (Ablation Study)
Comparison of different attention mechanisms integrated into the Norm-AImed framework:

| Attention Mechanism | mAP@0.5 (%) | Params (M) | Status |
| :--- | :---: | :---: | :--- |
| SE (Squeeze-and-Excitation) | 97.34% | 3.016 | Baseline+ |
| ECA (Efficient Channel) | 97.88% | 3.011 | Strong |
| CA (Coordinate Attention) | 95.57% | 3.018 | Mid |
| **EMA (Our Choice)** | **97.96%** | **3.022** | **SOTA** |
| GAM (Global Attention) | 94.50% | 3.154 | Heavy |
| CBAM (Convolutional Block) | 95.70% | 3.150 | Classic |

*Note: Benchmarked on RTX 5060 Ti at FP16 precision.*

---

## Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/Norm-AImed.git
cd Norm-AImed
pip install -r requirements.txt
```

### 2. Prepare Weights
The pre-trained EMA weights are located at `./ablation_runs/Exp_EMA/weights/best.pt`.

### 3. CLI Inference
Run detection on a single image or video:
```bash
python FINAL_LOBOTOMY.py --source ./demo/sample_polyp.jpg --weights ./ablation_runs/Exp_EMA/weights/best.pt
```

### 4. Training / Ablation
To reproduce the ablation studies:
```bash
python run_ablation_batch.py
```

---

## Acknowledgments
The success of this project is deeply rooted in the contributions of the open-source community and medical institutions. We express our sincere gratitude to the following organizations for providing critical endoscopic imaging data support for the YOLO architecture training:

### Open Source Dataset Support
* **Simula Research Laboratory & Bærum Hospital (Norway)**: For providing the **Kvasir-SEG** dataset, which offered high-resolution polyp images and precise segmentation masks.
* **Hospital Clínic de Barcelona & Computer Vision Center (Spain)**: For the **CVC-ClinicDB** and **CVC-ColonDB** (EndoScene) datasets. Their diverse lighting conditions and viewing angles significantly enhanced the model's generalization capabilities.
* **ETIS-Larib Laboratory (France)**: For the **ETIS-Larib Polyp DB**, which provided essential samples for detecting early-stage small polyps.
* **Showa University & Nagoya University (Japan)**: For the **SUN Database**, laying the foundation for robustness testing across multiple endoscopic platforms.

### Clinical & Academic Support
* Gratitude is extended to **[Your Partner Hospital Name]** for providing de-identified clinical data and expert annotation guidance. This support allowed the DCNv3 module to better adapt to real-world morphological features in clinical settings.

Additionally, special thanks to the Ultralytics team for the YOLOv8 foundation, and the contributors of WIoU and DCNv3 for their ground-breaking open-source work.

---
---

# Norm-AImed: 实时内窥镜息肉检测系统

## 摘要

**Norm-AImed** 是一款专为内窥镜手术设计的高性能、工业级实时息肉检测系统。该系统基于优化后的 **YOLOv8n** 骨干网络构建，集成了前沿的架构改进，实现了卓越的检测精度与极致的推理速度。

核心优化包括：

* **WIoU v3**: 动态非单调聚焦，用于提升边界框回归的鲁棒性。
* **BiFPN_Concat**: 高效的多尺度特征融合机制，增强对微小病灶的检测能力。
* **EMA Attention**: 跨空间坐标注意力机制，强化特征定位的精准度.
* **DCNv3**: 可变形卷积，动态适应息肉高度不规则的形态学特征。

系统在 NVIDIA RTX 5060 Ti 硬件环境下实现了 **225+ FPS** 的推理速度，并在标准结直肠公开数据集上达到了 **97.96%** 的 **mAP@0.5**。

---

## 核心特性

* 🚀 **极致的推理速度**: 在消费级 GPU 上达到 >225 FPS，提供无延迟的实时临床诊断辅助。
* 🎯 **高级注意力机制 (EMA)**: 通过利用跨空间坐标信息，在医学图像场景下的表现全面超越传统机制 (CBAM, SE, ECA)。
* 🧬 **形态适应性**: 深度集成 DCNv3，赋予模型动态调整感受野的能力，精准匹配多样的息肉几何结构。
* 🛡️ **防 OOM 稳定训练**: 实现异步 I/O 队列与内存优化算子，确保在有限显存环境下的训练稳定性。

---

## 性能矩阵 (消融实验)

Norm-AImed 框架内不同注意力机制的性能对比分析：

| 注意力机制 | mAP@0.5 (%) | 参数量 (M) | 状态 |
| :--- | :---: | :---: | :--- |
| SE (Squeeze-and-Excitation) | 97.34% | 3.016 | 基准提升 (Baseline+) |
| ECA (Efficient Channel) | 97.88% | 3.011 | 表现强劲 (Strong) |
| CA (Coordinate Attention) | 95.57% | 3.018 | 表现一般 (Mid) |
| **EMA (Our Choice)** | **97.96%** | **3.022** | **最佳状态 (SOTA)** |
| GAM (Global Attention) | 94.50% | 3.154 | 计算量大 (Heavy) |
| CBAM (Convolutional Block) | 95.70% | 3.150 | 经典方案 (Classic) |

*注：以上基准测试均在 RTX 5060 Ti 环境下以 FP16 精度运行。*

---

## 快速开始

### 1. 环境安装

克隆仓库并安装相关依赖项：

```bash
git clone https://github.com/your-username/Norm-AImed.git
cd Norm-AImed
pip install -r requirements.txt
```

### 2. 准备权重文件

预训练的 EMA 最佳权重文件路径为：`./ablation_runs/Exp_EMA/weights/best.pt`。

### 3. 命令行推理

对单张图像或视频进行推理检测：

```bash
python FINAL_LOBOTOMY.py --source ./demo/sample_polyp.jpg --weights ./ablation_runs/Exp_EMA/weights/best.pt
```

### 4. 训练与消融实验复现

执行以下脚本批量跑通消融实验：

```bash
python run_ablation_batch.py
```

---

## 致谢 (Acknowledgments)

本项目的成功离不开开源社区与医疗机构在高质量数据集建设上的贡献。特别鸣谢以下机构与团队为本项目 YOLO 架构的模型训练提供了至关重要的肠道内窥镜影像数据支持：

### 开源数据集支持
* **Simula Research Laboratory & Bærum Hospital (挪威)**: 感谢其提供的 **Kvasir-SEG** 数据集，为模型提供了大量高分辨率的胃肠道息肉图像及精准的掩码标注。
* **Hospital Clínic de Barcelona & Computer Vision Center (西班牙)**: 感谢其提供的 **CVC-ClinicDB** 与 **CVC-ColonDB (EndoScene)** 数据集，其涵盖的不同光源和视角的结肠镜影像大幅提升了模型的泛化能力。
* **ETIS-Larib 实验室 (法国)**: 感谢其提供的 **ETIS-Larib Polyp DB**，补充了早期微小息肉的检测样本。
* **Showa University & Nagoya University (日本)**: 感谢其建立的 **SUN Database**，为模型在多源肠镜设备下的鲁棒性测试提供了基础。

### 临床与学术支持
* 感谢 **[请替换为您合作的医院名称]** 提供的私有临床去敏数据与专业标注指导，使本系统的 DCNv3 模块能更好地适应真实的临床形态学特征。

此外，特别鸣谢 Ultralytics 团队提供的 YOLOv8 基础架构，以及 WIoU 和 DCNv3 项目所有开源贡献者的工作。
