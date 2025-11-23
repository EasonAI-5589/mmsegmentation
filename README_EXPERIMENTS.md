# 语义分割 Baseline 实验说明

## 📋 实验概览

本项目包含 6 个 baseline 实验，测试 3 种模型在 2 个数据集上的性能。

### 模型
1. **UNet** - 经典 CNN baseline
2. **SegFormer-B2** - Transformer 架构
3. **Mask2Former** - SOTA 全景分割模型

### 数据集
1. **Cityscapes** - 城市街景，19 类
2. **ADE20K** - 通用场景，150 类

---

## 🚀 快速开始

### 环境要求
- Python 3.7+
- PyTorch 1.6+
- CUDA 11.0+
- 8 × GPU（推荐显存 ≥ 16GB）

### 安装依赖
```bash
# 基础依赖
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Mask2Former 额外依赖
pip install "mmdet>=3.0.0rc4"
```

### 数据集准备
```bash
# Cityscapes
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/

# ADE20K
data/ade/ADEChallengeData2016/
├── annotations/
│   ├── training/
│   └── validation/
└── images/
    ├── training/
    └── validation/
```

---

## 📝 实验脚本

### 方案 1: 只运行 UNet（推荐先测试）
```bash
# 给脚本添加执行权限
chmod +x run_unet_only.sh

# 运行 UNet 实验（约 12-16 小时）
./run_unet_only.sh
```

### 方案 2: 运行所有 6 个实验
```bash
# 给脚本添加执行权限
chmod +x run_experiments.sh

# 运行所有实验（约 40-50 小时）
./run_experiments.sh
```

### 方案 3: 单独运行某个实验
```bash
# 例: 只运行 SegFormer-B2 on Cityscapes
bash tools/dist_train.sh \
    configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py \
    8 \
    --work-dir work_dirs/segformer_b2_cityscapes
```

---

## 📊 实验清单

| 实验 | 模型 | 数据集 | 配置文件 | 预期 mIoU | 时间 |
|------|------|--------|----------|-----------|------|
| 1 | UNet | Cityscapes | [unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py](configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py) | 69.10 | 6-8h |
| 2 | FCN | ADE20K | [fcn_r50-d8_4xb4-160k_ade20k-512x512.py](configs/fcn/fcn_r50-d8_4xb4-160k_ade20k-512x512.py) | 35.94 | 6-8h |
| 3 | SegFormer-B2 | Cityscapes | [segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py](configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py) | 81.08 | 6-8h |
| 4 | SegFormer-B2 | ADE20K | [segformer_mit-b2_8xb2-160k_ade20k-512x512.py](configs/segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py) | 45.58 | 8-10h |
| 5 | Mask2Former | Cityscapes | [mask2former_r50_8xb2-90k_cityscapes-512x1024.py](configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py) | 80.44 | 5-7h |
| 6 | Mask2Former | ADE20K | [mask2former_r50_8xb2-160k_ade20k-512x512.py](configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py) | 47.87 | 8-10h |

**总训练时间**: 约 40-50 小时

---

## 🔍 监控训练

### 查看日志
```bash
# 实时查看训练日志
tail -f work_dirs/unet_cityscapes/20231123_*.log

# 查看所有日志
ls -lh experiment_logs_*/
```

### 使用 TensorBoard
```bash
tensorboard --logdir work_dirs/
```

### 检查训练进度
```bash
# 查看 checkpoint 文件
ls -lh work_dirs/unet_cityscapes/*.pth

# 查看最新的 mIoU
grep -i "miou" work_dirs/unet_cityscapes/*.log | tail -5
```

---

## 📈 查看结果

### 训练完成后
```bash
# 所有实验的日志都在这里
ls experiment_logs_*/

# 查看某个实验的 mIoU
grep -i "miou" experiment_logs_*/exp1_unet_cityscapes.log
```

### 测试模型
```bash
# 测试某个训练好的模型
bash tools/dist_test.sh \
    configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py \
    work_dirs/unet_cityscapes/iter_160000.pth \
    8
```

---

## ⚠️ 常见问题

### 1. CUDA Out of Memory
```bash
# 解决方案：减小 batch size（修改配置文件）
# 或使用梯度累积
--cfg-options train_dataloader.batch_size=2
```

### 2. 训练中断后恢复
```bash
# 使用 --resume 参数
bash tools/dist_train.sh \
    configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py \
    8 \
    --resume \
    --work-dir work_dirs/unet_cityscapes
```

### 3. UNet on ADE20K 配置缺失
- 官方没有提供 UNet ADE20K 配置
- 使用 FCN (configs/fcn/fcn_r50-d8_4xb4-160k_ade20k-512x512.py) 作为替代
- FCN 是类似的全卷积网络架构

### 4. Mask2Former 依赖问题
```bash
# 先安装 mmdet
pip install "mmdet>=3.0.0rc4"
```

---

## 📂 文件结构

```
mmsegmentation/
├── configs/                    # 配置文件
│   ├── unet/
│   ├── segformer/
│   ├── mask2former/
│   └── fcn/
├── work_dirs/                  # 训练输出（自动生成）
│   ├── unet_cityscapes/
│   ├── fcn_ade20k/
│   ├── segformer_b2_cityscapes/
│   └── ...
├── experiment_logs_*/          # 实验日志（自动生成）
├── run_unet_only.sh            # UNet 实验脚本
├── run_experiments.sh          # 完整实验脚本
└── README_EXPERIMENTS.md       # 本文件
```

---

## 🎯 推荐执行流程

### 第一步：快速验证环境
```bash
# 使用预训练模型测试（约 5 分钟）
bash tools/dist_test.sh \
    configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py \
    https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth \
    8
```

### 第二步：运行 UNet baseline
```bash
# 运行 UNet 实验（约 12-16 小时）
chmod +x run_unet_only.sh
./run_unet_only.sh
```

### 第三步：运行完整实验
```bash
# 运行所有 6 个实验（约 40-50 小时）
chmod +x run_experiments.sh
./run_experiments.sh
```

---

## 📧 联系方式

如有问题，请提交 Issue 或联系维护者。

---

## 📚 参考资料

- [MMSegmentation 官方文档](https://mmsegmentation.readthedocs.io/)
- [SegFormer 论文](https://arxiv.org/abs/2105.15203)
- [Mask2Former 论文](https://arxiv.org/abs/2112.01527)
- [UNet 论文](https://arxiv.org/abs/1505.04597)
