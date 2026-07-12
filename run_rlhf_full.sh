#!/bin/bash
# run_rlhf_full.sh — 完整 RLHF 三阶段验证
# 每阶段只跑少量步数，验证代码能跑通
set -e

PROJECT="/home/hyl/project/RLHF_train"
export PYTHONPATH="/home/hyl/miniconda3/envs/rlhf/lib/python3.10/site-packages:$PYTHONPATH"

# 验证步数（通过环境变量控制）
# 设 SFT_STEPS/RM_STEPS/GRPO_STEPS 为步数模式（用于快速验证）
# 不设则为 epoch 模式（完整训练）
export SFT_STEPS=${SFT_STEPS:-}     # 留空 = epoch模式
export RM_STEPS=${RM_STEPS:-}       # 留空 = epoch模式
export GRPO_STEPS=${GRPO_STEPS:-3}  # GRPO 默认 3 步

echo "========================================"
echo "  RLHF 三阶段全参数训练"
echo "========================================"
echo "硬件: NVIDIA GB10 (121GB 统一内存)"
echo "模型: Qwen2.5-1.5B (1.5B 全参数)"
echo "数据: Anthropic/hh-rlhf (完整数据集)"
echo "  SFT: 159K 条 | RM: 160K 条 | PPO: 50 prompts"
echo ""
if [ -z "$SFT_STEPS" ] && [ -z "$RM_STEPS" ]; then
    echo "模式: EPOCH 模式 (完整训练)"
    echo "  SFT EPOCHS: ${SFT_EPOCHS:-1}"
    echo "  RM  EPOCHS: ${RM_EPOCHS:-1}"
    echo "  GRPO STEPS: ${GRPO_STEPS:-3}"
else
    echo "模式: STEP 模式 (快速验证)"
    echo "  SFT: ${SFT_STEPS:-epoch} 步"
    echo "  RM:  ${RM_STEPS:-epoch} 步"
    echo "  GRPO: ${GRPO_STEPS:-3} 步"
fi
echo "========================================"
echo ""

# 数据检查
SFT_DATA="$PROJECT/data/sft_data.json"
RM_DATA="$PROJECT/data/rm_data.json"
PPO_DATA="$PROJECT/data/ppo_prompts.json"
MODEL="$PROJECT/models/qwen2.5-1.5b/model.safetensors"

for f in "$SFT_DATA" "$RM_DATA" "$PPO_DATA"; do
    if [ ! -f "$f" ]; then
        echo "⚠️  数据缺失: $f，先运行数据准备..."
        source /home/hyl/miniconda3/bin/activate rlhf
        python "$PROJECT/scripts/prepare_data.py"
        break
    fi
done

if [ ! -f "$MODEL" ]; then
    echo "❌ 模型文件缺失！"
    exit 1
fi

# ===== Stage 1: SFT =====
echo ""
echo "████████████████████████████████████████████████"
echo "█  Stage 1: SFT 全参数微调 (${SFT_STEPS} 步)"
echo "████████████████████████████████████████████████"
source /home/hyl/miniconda3/bin/activate rlhf
python "$PROJECT/scripts/stage1_sft.py"
echo "✅ Stage 1 完成！"

# ===== Stage 2: RM =====
echo ""
echo "████████████████████████████████████████████████"
echo "█  Stage 2: Reward Model 训练 (${RM_STEPS} 步)"
echo "████████████████████████████████████████████████"
python "$PROJECT/scripts/stage2_rm.py"
echo "✅ Stage 2 完成！"

# ===== Stage 3: GRPO =====
echo ""
echo "████████████████████████████████████████████████"
echo "█  Stage 3: GRPO 强化学习 (${GRPO_STEPS} 步)"
echo "████████████████████████████████████████████████"
python "$PROJECT/scripts/stage3_grpo.py"
echo "✅ Stage 3 完成！"

echo ""
echo "========================================"
echo "  🎉 RLHF 三阶段代码验证全部通过！"
echo "========================================"
echo ""
echo "产出文件:"
echo "  SFT:  $PROJECT/output/sft/"
echo "  RM:   $PROJECT/output/rm/"
echo "  GRPO: $PROJECT/output/grpo/"
echo ""
echo "说明: 每阶段只跑了少量步数用于验证代码"
echo "完整训练需要移除环境变量限制或调大步数"
echo "========================================"
