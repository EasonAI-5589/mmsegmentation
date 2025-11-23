#!/bin/bash

# ========================================
# 单独运行 UNet 实验脚本（8卡配置）
# ========================================

set -e  # 遇到错误立即退出

# 配置
GPUS=8
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="unet_logs_${TIMESTAMP}"

# 创建日志目录
mkdir -p ${LOG_DIR}

echo "========================================"
echo "UNet Baseline 实验"
echo "时间: ${TIMESTAMP}"
echo "GPU数量: ${GPUS}"
echo "日志目录: ${LOG_DIR}"
echo "========================================"

# ========================================
# 实验 1: UNet on Cityscapes
# ========================================
echo ""
echo "[实验 1/2] UNet on Cityscapes"
echo "----------------------------------------"
echo "配置: configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py"
echo "预期 mIoU: ~69.10"
echo "训练轮数: 160k iterations"
echo "预计时间: 6-8 小时"
echo ""

bash tools/dist_train.sh \
    configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py \
    ${GPUS} \
    --work-dir work_dirs/unet_cityscapes \
    2>&1 | tee ${LOG_DIR}/unet_cityscapes.log

echo ""
echo "✅ UNet on Cityscapes 训练完成"

# 测试
echo ""
echo "开始测试 UNet on Cityscapes..."
bash tools/dist_test.sh \
    configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py \
    work_dirs/unet_cityscapes/iter_160000.pth \
    ${GPUS} \
    2>&1 | tee ${LOG_DIR}/unet_cityscapes_test.log

# ========================================
# 实验 2: FCN (UNet替代) on ADE20K
# ========================================
echo ""
echo "[实验 2/2] FCN (UNet替代) on ADE20K"
echo "----------------------------------------"
echo "配置: configs/fcn/fcn_r50-d8_4xb4-160k_ade20k-512x512.py"
echo "预期 mIoU: ~35.94"
echo "训练轮数: 160k iterations"
echo "预计时间: 6-8 小时"
echo "⚠️  注意: UNet 没有官方 ADE20K 配置，使用 FCN 作为替代"
echo ""

bash tools/dist_train.sh \
    configs/fcn/fcn_r50-d8_4xb4-160k_ade20k-512x512.py \
    ${GPUS} \
    --work-dir work_dirs/fcn_ade20k \
    2>&1 | tee ${LOG_DIR}/fcn_ade20k.log

echo ""
echo "✅ FCN on ADE20K 训练完成"

# 测试
echo ""
echo "开始测试 FCN on ADE20K..."
bash tools/dist_test.sh \
    configs/fcn/fcn_r50-d8_4xb4-160k_ade20k-512x512.py \
    work_dirs/fcn_ade20k/iter_160000.pth \
    ${GPUS} \
    2>&1 | tee ${LOG_DIR}/fcn_ade20k_test.log

# ========================================
# 汇总结果
# ========================================
echo ""
echo "========================================"
echo "🎉 UNet 实验完成！"
echo "========================================"
echo ""
echo "📊 结果汇总："
echo "----------------------------------------"

echo ""
echo "1️⃣  UNet on Cityscapes"
echo "训练日志: ${LOG_DIR}/unet_cityscapes.log"
echo "测试日志: ${LOG_DIR}/unet_cityscapes_test.log"
grep -i "miou" ${LOG_DIR}/unet_cityscapes_test.log | tail -3 || echo "  (请查看日志文件)"

echo ""
echo "2️⃣  FCN on ADE20K"
echo "训练日志: ${LOG_DIR}/fcn_ade20k.log"
echo "测试日志: ${LOG_DIR}/fcn_ade20k_test.log"
grep -i "miou" ${LOG_DIR}/fcn_ade20k_test.log | tail -3 || echo "  (请查看日志文件)"

echo ""
echo "📁 详细日志目录: ${LOG_DIR}"
echo "💾 模型权重目录: work_dirs/"
echo ""
echo "下一步："
echo "  - 查看训练曲线: tensorboard --logdir work_dirs/"
echo "  - 运行其他模型: bash run_experiments.sh"
echo ""
