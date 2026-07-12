#!/bin/bash
# run_rlhf_pipeline.sh — RLHF 训练流水线
set -e

source /home/hyl/miniconda3/bin/activate rlhf

PROJECT_DIR="/home/hyl/project/RLHF_train"

echo "========================================"
echo "  RLHF (DPO) 训练流水线 - Qwen2.5-1.5B"
echo "========================================"

# 检查模型是否下载完成
MODEL_PATH="$PROJECT_DIR/models/qwen2.5-1.5b"
if [ ! -f "$MODEL_PATH/model.safetensors" ]; then
    echo "❌ 模型文件未找到: $MODEL_PATH/model.safetensors"
    echo "请先下载模型"
    exit 1
fi
echo "✅ 模型文件已就绪"

MODEL_SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
echo "模型大小: $MODEL_SIZE"

# 检查数据
DATA_PATH="$PROJECT_DIR/data/preference_data.json"
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 数据文件未找到"
    exit 1
fi
DATA_COUNT=$(python3 -c "import json; print(len(json.load(open('$DATA_PATH'))))")
echo "✅ 偏好数据已就绪: $DATA_COUNT 条"

echo ""
echo "========================================"
echo "  开始 DPO 训练..."
echo "========================================"
echo "硬件: NVIDIA GB10 (121GB 统一内存)"
echo "模型: Qwen2.5-1.5B-Instruct (1.5B)"
echo "数据: $DATA_COUNT 条偏好对"
echo "方法: QLoRA (4-bit NF4) + DPO"
echo ""
echo "⏱️ 预估训练时间:"
echo "  每条训练步骤: ~5-10秒"
echo "  训练步骤: $(( DATA_COUNT / 8 * 3 )) 步 (batch=8, epoch=3)"
echo "  总时间: ~5-10分钟"
echo "========================================"
echo ""

python3 "$PROJECT_DIR/scripts/train_dpo.py"

echo ""
echo "========================================"
echo "  训练完成！运行推理测试..."
echo "========================================"
echo ""

python3 "$PROJECT_DIR/scripts/inference_test.py"

echo ""
echo "========================================"
echo "  流水线执行完毕！"
echo "========================================"
