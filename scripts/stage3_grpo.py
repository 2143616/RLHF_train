"""
Stage 3: GRPO (Group Relative Policy Optimization) — RLHF 第三阶段
使用 GRPO 优化策略，提高回答质量
使用 rule-based reward functions 作为验证
"""
import os, json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# ===== 配置 =====
MODEL_PATH = "/home/hyl/project/RLHF_train/output/sft"  # 用 SFT 后的模型
DATA_PATH = "/home/hyl/project/RLHF_train/data/ppo_prompts.json"
OUTPUT_DIR = "/home/hyl/project/RLHF_train/output/grpo"
MAX_STEPS = int(os.environ.get("GRPO_STEPS", "3"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print(f"Stage 3: GRPO — RLHF 强化学习优化")
print(f"最大步数: {MAX_STEPS}")
print("=" * 55)

# 1. tokenizer
print("\n[1/5] 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. 加载 SFT 后的模型
print("\n[2/5] 加载 SFT 模型（全参数 bf16）...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  总参数量: {model.num_parameters():,}")

# 3. 加载 prompts
print("\n[3/5] 加载 prompts...")
with open(DATA_PATH) as f:
    raw = json.load(f)
# 只取前 10 条用于验证
prompts = raw[:10]
dataset_dict = {
    "prompt": [item["prompt"] for item in prompts],
    # GRPOTrainer 还可以接受额外的 reward 相关字段
}
dataset = Dataset.from_dict(dataset_dict)
print(f"  训练 prompts: {len(dataset)} 条")

# 4. 定义 reward functions
print("\n[4/5] 定义 reward functions...")

def reward_length(completions, **kwargs):
    """鼓励详细回答（长度奖励）"""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) and len(c) > 0 else str(c)
        length = len(text)
        # 太长或太短都不好，50-200 字最佳
        if length < 20:
            r = -0.5
        elif length < 50:
            r = 0.0
        elif length < 200:
            r = 0.5
        else:
            r = 0.3
        rewards.append(r)
    return rewards

def reward_format(completions, **kwargs):
    """奖励结构性回答（含列举、代码块、标题等）"""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) and len(c) > 0 else str(c)
        r = 0.0
        if "1." in text or "- " in text or "•" in text:
            r += 0.2
        if "**" in text or "###" in text or "##" in text:
            r += 0.2
        if "```" in text:
            r += 0.3
        if text.endswith(("?", "!", ".")):
            r += 0.1
        rewards.append(r)
    return rewards

def reward_safety(completions, **kwargs):
    """安全奖励：拒绝危险内容的回答加分"""
    rewards = []
    danger_words = ["bomb", "kill", "steal", "drug", "illegal", "hack",
                    "weapon", "poison", "die", "suicide"]
    safe_phrases = ["cannot", "can't", "sorry", "unable", "not appropriate",
                    "illegal", "against", "harmful", "dangerous"]
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) and len(c) > 0 else str(c).lower()
        r = 0.0
        has_danger = any(w in text.lower() for w in danger_words)
        has_safe_refusal = any(p in text.lower() for p in safe_phrases)
        if has_danger and has_safe_refusal:
            r = 1.0  # 遇到危险请求但正确拒绝
        elif has_danger and not has_safe_refusal:
            r = -1.0  # 危险请求没有拒绝
        else:
            r = 0.2  # 安全话题
        rewards.append(r)
    return rewards

reward_funcs = [reward_length, reward_format, reward_safety]
print(f"  定义了 {len(reward_funcs)} 个 reward functions:")
for fn in reward_funcs:
    print(f"    - {fn.__name__}")

# 5. 配置 GRPO
print("\n[5/5] 配置 GRPO Trainer...")
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-6,
    max_steps=MAX_STEPS,
    logging_steps=1,
    save_steps=MAX_STEPS,
    save_total_limit=1,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    dataloader_num_workers=1,
    num_generations=4,           # 每组生成 4 个回答
    max_completion_length=256,
    beta=0.04,                   # KL 惩罚系数
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print(f"\n  训练步数: {MAX_STEPS}")
print(f"  num_generations: {training_args.num_generations}")
print(f"  beta (KL): {training_args.beta}")

# 6. 训练
print("\n" + "=" * 55)
print(f"开始 GRPO 训练 ({MAX_STEPS} 步)...")
print("=" * 55)
trainer.train()

# 7. 保存
print("\n保存 GRPO 模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ GRPO 完成！模型已保存到 {OUTPUT_DIR}")
