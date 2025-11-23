#!/bin/bash

# ========================================
# 语义分割 Baseline 实验脚本（8卡配置）
# ========================================

set -e  # 遇到错误立即退出

# 配置
GPUS=8
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="experiment_logs_${TIMESTAMP}"

# 创建日志目录
mkdir -p ${LOG_DIR}

echo "========================================"
echo "开始 Baseline 实验"
echo "时间: ${TIMESTAMP}"
echo "GPU数量: ${GPUS}"
echo "日志目录: ${LOG_DIR}"
echo "========================================"

# ========================================
# 实验 1: UNet on Cityscapes
# ========================================
echo ""
echo "[实验 1/6] UNet on Cityscapes"
echo "----------------------------------------"
bash tools/dist_train.sh \
    configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py \
    ${GPUS} \
    --work-dir work_dirs/exp1_unet_cityscapes \
    2>&1 | tee ${LOG_DIR}/exp1_unet_cityscapes.log

echo "✅ 实验 1 完成"

# ========================================
# 实验 2: UNet on ADE20K (使用 FCN 替代)
# ========================================
echo ""
echo "[实验 2/6] FCN (UNet替代) on ADE20K"
echo "----------------------------------------"
bash tools/dist_train.sh \
    configs/fcn/fcn_r50-d8_4xb4-160k_ade20k-512x512.py \
    ${GPUS} \
    --work-dir work_dirs/exp2_fcn_ade20k \
    2>&1 | tee ${LOG_DIR}/exp2_fcn_ade20k.log

echo "✅ 实验 2 完成"

# ========================================
# 实验 3: SegFormer-B2 on Cityscapes
# ========================================
echo ""
echo "[实验 3/6] SegFormer-B2 on Cityscapes"
echo "----------------------------------------"
bash tools/dist_train.sh \
    configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py \
    ${GPUS} \
    --work-dir work_dirs/exp3_segformer_b2_cityscapes \
    2>&1 | tee ${LOG_DIR}/exp3_segformer_b2_cityscapes.log

echo "✅ 实验 3 完成"

# ========================================
# 实验 4: SegFormer-B2 on ADE20K
# ========================================
echo ""
echo "[实验 4/6] SegFormer-B2 on ADE20K"
echo "----------------------------------------"
bash tools/dist_train.sh \
    configs/segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py \
    ${GPUS} \
    --work-dir work_dirs/exp4_segformer_b2_ade20k \
    2>&1 | tee ${LOG_DIR}/exp4_segformer_b2_ade20k.log

echo "✅ 实验 4 完成"

# ========================================
# 实验 5: Mask2Former on Cityscapes
# ========================================
echo ""
echo "[实验 5/6] Mask2Former on Cityscapes"
echo "----------------------------------------"
echo "检查 mmdet 依赖..."
python -c "import mmdet; print(f'MMDetection version: {mmdet.__version__}')" 2>/dev/null || {
    echo "⚠️  mmdet 未安装，正在安装..."
    pip install "mmdet>=3.0.0rc4"
}

bash tools/dist_train.sh \
    configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py \
    ${GPUS} \
    --work-dir work_dirs/exp5_mask2former_cityscapes \
    2>&1 | tee ${LOG_DIR}/exp5_mask2former_cityscapes.log

echo "✅ 实验 5 完成"

# ========================================
# 实验 6: Mask2Former on ADE20K
# ========================================
echo ""
echo "[实验 6/6] Mask2Former on ADE20K"
echo "----------------------------------------"
bash tools/dist_train.sh \
    configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py \
    ${GPUS} \
    --work-dir work_dirs/exp6_mask2former_ade20k \
    2>&1 | tee ${LOG_DIR}/exp6_mask2former_ade20k.log

echo "✅ 实验 6 完成"

# ========================================
# 汇总结果
# ========================================
echo ""
echo "========================================"
echo "🎉 所有实验完成！"
echo "========================================"
echo ""
echo "实验结果汇总："
echo "----------------------------------------"

for log in ${LOG_DIR}/*.log; do
    echo "📄 $(basename $log)"
    grep -i "miou" $log | tail -3 || echo "  (训练中或日志未找到 mIoU)"
    echo ""
done

echo "详细日志目录: ${LOG_DIR}"
echo "模型权重目录: work_dirs/"
echo ""
