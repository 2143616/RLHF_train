"""
Stage 3: GRPO (Group Relative Policy Optimization) — RLHF 第三阶段
使用 Stage 2 训练的 Reward Model 评分，优化策略提高回答质量
"""
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# ===== 配置 =====
MODEL_PATH = "/home/hyl/project/RLHF_train/output/sft"      # SFT 后的策略模型
RM_MODEL_PATH = "/home/hyl/project/RLHF_train/output/rm"     # Stage 2 训练的奖励模型
DATA_PATH = "/home/hyl/project/RLHF_train/data/ppo_prompts.json"
OUTPUT_DIR = "/home/hyl/project/RLHF_train/output/grpo"
MAX_STEPS = int(os.environ.get("GRPO_STEPS", "3"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print(f"Stage 3: GRPO — RLHF 强化学习优化（RM 模型评分）")
print(f"最大步数: {MAX_STEPS}")
print("=" * 55)

# 1. tokenizer（SFT 模型用）
print("\n[1/6] 加载 SFT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. 加载 SFT 后的策略模型
print("\n[2/6] 加载 SFT 策略模型（全参数 bf16）...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  策略模型参数量: {model.num_parameters():,}")

# 3. 加载 RM 奖励模型
print("\n[3/6] 加载 RM 奖励模型...")
rm_model = AutoModelForCausalLM.from_pretrained(
    RM_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(RM_MODEL_PATH, trust_remote_code=True)
rm_tokenizer.pad_token = rm_tokenizer.eos_token
rm_model.eval()
print(f"  RM 模型参数量: {rm_model.num_parameters():,}")

# 4. 加载 prompts
print("\n[4/6] 加载 prompts...")
with open(DATA_PATH) as f:
    raw = json.load(f)
prompts = raw[:10]  # 只取前 10 条用于验证
dataset_dict = {
    "prompt": [item["prompt"] for item in prompts],
}
dataset = Dataset.from_dict(dataset_dict)
print(f"  训练 prompts: {len(dataset)} 条")

# 5. 定义 RM 评分函数
print("\n[5/6] 定义奖励函数（RM 模型评分）...")


def reward_rm_score(completions, prompts, **kwargs):
    """使用 Stage 2 训练的 RM 模型对生成回答打分

    completions: list[list[dict]], 每组回答 [{"role": "assistant", "content": "..."}]
    prompts: list[str], 原始 prompt
    返回: list[float], 每个回答的奖励分数
    """
    texts = []
    for prompt, completion in zip(prompts, completions):
        # 提取回答文本
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", str(completion))
        else:
            content = str(completion)

        # 格式化完整对话（与 RM 训练时一致）
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content},
        ]
        text = rm_tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    # Tokenize
    inputs = rm_tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=1024,
    )
    inputs = {k: v.to(rm_model.device) for k, v in inputs.items()}

    # RM 推理
    with torch.no_grad():
        outputs = rm_model(**inputs)
        logits = outputs.logits  # [batch_size, seq_len, 1]

    # 取每个样本最后一个非 padding token 的分数作为奖励
    last_indices = inputs["attention_mask"].sum(dim=1) - 1  # [batch_size]
    rewards = logits[torch.arange(len(texts)), last_indices, 0].tolist()

    return rewards


reward_funcs = [reward_rm_score]
print(f"  奖励函数: reward_rm_score（RM 模型）")

# 6. 配置 GRPO
print("\n[6/6] 配置 GRPO Trainer...")
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

# 7. 训练
print("\n" + "=" * 55)
print(f"开始 GRPO 训练 ({MAX_STEPS} 步)...")
print("=" * 55)
trainer.train()

# 8. 保存
print("\n保存 GRPO 模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ GRPO 完成！模型已保存到 {OUTPUT_DIR}")
