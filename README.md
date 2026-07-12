# RLHF 三阶段全参数训练项目

完整的 **SFT → RM → GRPO** 三阶段 RLHF 训练流程，使用 **Qwen2.5-1.5B** 全参数微调。

## 项目结构

```
RLHF_train/
├── data/
│   ├── sft_data.json          # SFT 指令数据 (500条)
│   ├── rm_data.json           # RM 偏好数据 (500条)
│   └── ppo_prompts.json       # GRPO prompts (50条)
├── models/qwen2.5-1.5b/      # Qwen2.5-1.5B 基座模型
├── scripts/
│   ├── prepare_data.py        # 从 Anthropic/hh-rlhf 提取数据
│   ├── stage1_sft.py          # Stage 1: 监督微调
│   ├── stage2_rm.py           # Stage 2: 奖励模型
│   └── stage3_grpo.py         # Stage 3: GRPO 强化学习
├── output/
│   ├── sft/                   # SFT 产出 (12GB 全参数)
│   ├── rm/                    # RM 产出 (12GB)
│   └── grpo/                  # GRPO 产出 (12GB)
├── run_rlhf_full.sh           # 一键三阶段流水线
└── README.md
```

## 数据集

数据来自 **Anthropic/hh-rlhf**（真实人类偏好数据集，16 万条），从 hf-mirror 加载。

| 子集 | 数量 | 用途 |
|------|------|------|
| SFT 指令 | 500 条 | 让模型学会遵循指令格式 |
| RM 偏好对 | 500 条 | 训练奖励模型区分好坏回答 |
| GRPO prompts | 50 条 | 强化学习策略优化 |

## 三阶段流程

### Stage 1: SFT（监督微调）
- 全参数 bf16 微调（15亿参数全部训练）
- 使用 HuggingFace Trainer
- 学习率 2e-5, batch_size 8
- **验证结果** (5步): Loss 3.56 → 2.22

### Stage 2: RM（奖励模型）
- 使用 RewardTrainer, num_labels=1
- 学习率 1e-5
- **验证结果** (5步): Mean reward -0.47 → +0.26

### Stage 3: GRPO（强化学习）
- 使用 GRPOTrainer（替代传统 PPO）
- 3 个 rule-based reward functions
- num_generations=4, beta=0.04
- **验证结果** (3步): KL 0.001, Reward ~0.28

## 环境要求

```bash
conda create -n rlhf python=3.10 -y
conda activate rlhf

# PyTorch (aarch64 NVIDIA GB10)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 训练框架
pip install transformers datasets accelerate peft bitsandbytes trl pillow
```

## 运行

```bash
# 环境变量控制步数（移除即完整训练）
export SFT_STEPS=5 RM_STEPS=5 GRPO_STEPS=3

# 一键三阶段
bash run_rlhf_full.sh

# 或单阶段运行
python scripts/stage1_sft.py
python scripts/stage2_rm.py
python scripts/stage3_grpo.py
```

## 硬件

- NVIDIA GB10 (ARM64, CC 12.1, 121GB 统一内存)
- 全参数 1.5B 模型: 约 12GB（权重）+ 6GB（优化器）= 约 20GB
- 建议 24GB+ 显存进行完整训练

## 参考

- [TRL 文档](https://huggingface.co/docs/trl)
- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [GRPO 论文](https://arxiv.org/abs/2402.03300)
