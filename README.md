# HHW-Ego Project Source Code

> 高光谱引导的可穿戴设备图像增强 - 源代码仓库

---

## 📁 项目结构

```
source_code/
├── realesrgan/           # 第一阶段: Real-ESRGAN超分辨率
│   ├── archs/            # 网络架构
│   ├── data/             # 数据加载器
│   ├── models/           # 模型训练/推理
│   ├── train.py          # 训练入口
│   └── inference_realesrgan.py  # 推理脚本
│
├── hsi/                # 第二阶段: 高光谱处理
│   ├── hsi_pipeline.py                    # 基础HSI流水线
│   ├── hsi_pipeline_window_mask_overexposure.py  # 高级色彩校正(推荐)
│   ├── hsi_to_xyz.py                     # HSI→XYZ转换
│   ├── xyz_to_cct.py                     # XYZ→色温转换
│   ├── hsi_rgb_alignment.py              # HSI-RGB对齐
│   └── hsi_visualization.py              # 可视化工具
│
└── experiments/         # 实验结果和配置
```

---

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
conda create -n hsi_env python=3.9 -y
conda activate hsi_env

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install basicsr opencv-python numpy matplotlib scikit-image scipy pillow colour-science
pip install pyyaml tensorboard
```

### 数据路径配置

根据你的数据集位置，需要在代码中配置以下路径：

```python
# 数据集根目录 (修改为你实际的路径)
DATASET_ROOT = "/Volumes/PSSD/hyperspectral/HSI_dataset/"

# 各数据子目录
GT_DIR = f"{DATASET_ROOT}/HDR_img/"
LQ_DIR = f"{DATASET_ROOT}/Glasses_img/"
HSI_DIR = f"{DATASET_ROOT}/HSI_data/"
RAYNEO_DIR = f"{DATASET_ROOT}/RayNeo_img/"
EMDOOR_DIR = f"{DATASET_ROOT}/EmdoorVR_img/"
```

---

## 📊 实验计划

详见: [`../EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md)

### 需要重跑的实验 (论文现有)

1. **Table 2**: SOTA方法对比
   - 基线: BSRGAN, Real-ESRGAN, StableSR, PASD, SUPIR, PFT-SR
   - 我们的方法: Ours (Real-ESRGAN/PASD/SUPIR/PFT based)

2. **Table 3**: 模块消融实验
   - Original wearable
   - Hyperspectral enhancement
   - SR enhancement
   - SR + Hyperspectral
   - SR + Hyperspectral + Exposure

3. **Table 4**: 退化配置消融
   - random-random-random
   - align-random-random
   - align-kernel-random
   - align-kernel-align

### 需要补充的实验 (审稿意见)

- **E1**: Strength参数消融实验
- **E2**: 颜色专用指标 (CIEDE2000, ΔE)
- **E3**: Lab对齐 vs Color Jittering
- **E4**: MUSIQ无参考指标
- **E5**: 配对测试集验证

---

## 🔄 Git工作流程

### 初始提交 (当前状态)

```bash
# 查看当前状态
git status

# 添加所有源代码
git add .

# 首次提交
git commit -m "Initial commit: Baseline code before re-experiments

- Real-ESRGAN module in realesrgan/
- HSI processing module in hsi/
- Original training and inference scripts
- This marks the starting point for all re-experiments"
```

### 实验分支策略

```bash
# 创建主开发分支
git checkout -b develop

# 创建各实验功能分支
git checkout -b feature-table2-sota-comparison
git checkout -b feature-table3-module-ablation
git checkout -b feature-table4-degradation-ablation

# 创建审稿意见实验分支
git checkout -b feature-E1-strength-ablation
git checkout -b feature-E2-color-metrics
git checkout -b feature-E3-lab-vs-jitter
git checkout -b feature-E4-musiq
git checkout -b feature-E5-paired-testset
```

### 提交规范

```bash
# 格式: <type>: <subject>

# 类型:
# exp: 实验运行和结果
# feat: 新功能实现
# fix: bug修复
# refactor: 代码重构
# docs: 文档更新

# 示例:
git commit -m "exp: Run Table 2 SOTA comparison"
git commit -m "feat: Implement CIEDE2000 metric"
git commit -m "exp: E1 strength ablation completed"
```

---

## 📝 关键文件说明

### 第一阶段: Real-ESRGAN

| 文件 | 功能 | 优先级 |
|------|------|--------|
| `realesrgan/train.py` | 训练入口 | ⭐⭐⭐ |
| `realesrgan/inference_realesrgan.py` | 推理脚本 | ⭐⭐⭐ |
| `realesrgan/options/train.yml` | 训练配置 | ⭐⭐ |
| `realesrgan/data/telehyper_dataset.py` | 数据加载器 | ⭐⭐ |
| `realesrgan/models/realesrnet_model.py` | 模型定义 | ⭐ |

### 第二阶段: HSI处理

| 文件 | 功能 | 优先级 |
|------|------|--------|
| `hsi/hsi_pipeline_window_mask_overexposure.py` | 核心HSI流水线(推荐) | ⭐⭐⭐ |
| `hsi/hsi_pipeline.py` | 基础HSI流水线 | ⭐⭐ |
| `hsi/hsi_rgb_alignment.py` | HSI-RGB对齐 | ⭐⭐ |
| `hsi/hsi_visualization.py` | 可视化工具 | ⭐ |
| `hsi/hsi_to_xyz.py` | 颜色空间转换 | ⭐ |

---

## 🔧 配置文件

### 训练配置: `realesrgan/options/train.yml`

需要根据新数据集调整的关键参数:

```yaml
train:
  name: finetune_RealESRGANx4plus
  scale: 4
  datasets:
    train:
      name: TeleHyperDataset
      # 修改为你的数据路径
      dataroot_gt: /path/to/HDR_img
      dataroot_lq: /path/to/Glasses_img
      meta_info: /path/to/meta_info.txt
```

### HSI参数配置

需要调优的关键参数 (hsi_pipeline_window_mask_overexposure.py):

```python
# 色彩饱和度增强
chroma_boost = 1.5  # 范围: 1.0-2.0

# 亮度动态范围
tone_mapping_max = 1.2  # 范围: 0.95-1.5

# CCT映射强度 (E1消融实验)
strength = 1.0  # E1测试: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
```

---

## 📊 实验结果目录

```bash
# 创建实验结果目录
mkdir -p experiments/results
mkdir -p experiments/figures
mkdir -p experiments/tables
mkdir -p experiments/logs
```

### 结果文件命名规范

```
experiments/
├── results/
│   ├── Table2_SOTA_comparison.csv
│   ├── Table3_ablation_modules.csv
│   ├── Table4_degradation_ablation.csv
│   ├── E1_strength_ablation.csv
│   ├── E2_color_metrics.csv
│   ├── E3_lab_vs_jitter.csv
│   ├── E4_musiq_results.csv
│   └── E5_paired_testset.csv
│
├── figures/
│   ├── comp-SOTA.png
│   ├── ablation-cameras2.png
│   ├── compare.png (色域分析)
│   ├── E1_strength_curves.png
│   └── E3_color_distribution.png
│
└── logs/
    ├── train_*.log
    └── exp_*.log
```

---

## 🎯 下一步行动

### 立即开始

```bash
# 1. 确认当前状态
git status

# 2. 提交当前基线
git add .
git commit -m "Initial: Baseline code before re-experiments"

# 3. 创建开发分支
git checkout -b develop

# 4. 开始第一个实验
# 建议: 从最简单的基线指标开始
```

### 第一个实验建议

建议按以下顺序开始实验：

1. **Day 1**: 配置数据路径，运行基线测试
2. **Day 2-3**: Table 2 - SOTA对比实验
3. **Day 4**: Table 3 - 模块消融实验
4. **Day 5**: Table 4 - 退化配置消融
5. **Day 6-10**: 补充实验E1-E5
6. **Day 11-12**: 结果汇总和图表生成
7. **Day 13-14**: 论文更新和最终检查

---

## 📖 相关文档

- **实验计划**: [`../EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md)
- **项目结构**: [`../README_PROJECT_STRUCTURE.md`](../README_PROJECT_STRUCTURE.md)
- **审稿意见**: [`../agent asserts/iclr-reviews.md`](../agent%20asserts/iclr-reviews.md)
- **改进计划**: [`../agent asserts/iclr-paper-improvement-plan-zh.md`](../agent%20asserts/iclr-paper-improvement-plan-zh.md)
- **论文**: [`../paper/iclr2026_conference.tex`](../paper/iclr2026_conference.tex)

---

## 🔗 外部依赖

- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- BasicSR: https://github.com/XPixelGroup/BasicSR
- ICLR 2026: https://iclr.cc/

---

**最后更新**: 2026-02-22
**当前版本**: v1.0-baseline
