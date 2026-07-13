"""
Stage 2: RM (Reward Model) 训练
适配完整 160K 偏好数据集
"""
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

MODEL_PATH = "/home/hyl/project/RLHF_train/models/qwen2.5-1.5b"
DATA_PATH = "/home/hyl/project/RLHF_train/data/rm_data.json"
OUTPUT_DIR = "/home/hyl/project/RLHF_train/output/rm"

MAX_STEPS = int(os.environ.get("RM_STEPS") or "0")
NUM_EPOCHS = float(os.environ.get("RM_EPOCHS") or "1")

if MAX_STEPS > 0:
    HF_MAX_STEPS = MAX_STEPS
    HF_NUM_EPOCHS = 0.0
else:
    HF_MAX_STEPS = -1
    HF_NUM_EPOCHS = NUM_EPOCHS

VALIDATION_SPLIT = min(int(os.environ.get("RM_VALID", "5000")), 10000)

os.makedirs(OUTPUT_DIR, exist_ok=True)

mode = f"{MAX_STEPS} 步" if MAX_STEPS > 0 else f"{NUM_EPOCHS} epoch"
print(f"=" * 55)
print(f"Stage 2: Reward Model | {mode}")
print(f"=" * 55)

# 1. Tokenizer
print("[1/4] 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. 模型
print("[2/4] 加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, num_labels=1,
)
print(f"  参数量: {model.num_parameters():,}")

# 3. 数据
print("[3/4] 加载并格式化偏好数据...")
ds = load_dataset("json", data_files=DATA_PATH, split="train", streaming=False)
print(f"  原始: {len(ds)} 条")

def format_dialog(prompt, response):
    return tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ], tokenize=False)

def format_batch(examples):
    return {
        "chosen": [format_dialog(p, c) for p, c in zip(examples["prompt"], examples["chosen"])],
        "rejected": [format_dialog(p, r) for p, r in zip(examples["prompt"], examples["rejected"])],
    }

ds = ds.map(format_batch, batched=True, batch_size=1000,
            remove_columns=["prompt", "chosen", "rejected"])
ds = ds.train_test_split(test_size=VALIDATION_SPLIT, seed=42)
print(f"  训练: {len(ds['train'])} 条, 验证: {len(ds['test'])} 条")

# 4. 训练
print("[4/4] 配置 RewardTrainer...")
args = RewardConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    max_steps=HF_MAX_STEPS,
    num_train_epochs=HF_NUM_EPOCHS,
    logging_steps=10,
    save_steps=500, save_total_limit=2,
    eval_strategy="steps", eval_steps=500,
    bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",
    remove_unused_columns=False,
    report_to="none", dataloader_num_workers=2,
)

trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
)

print(f"\n开始训练...")
try:
    trainer.train()
except KeyboardInterrupt:
    print("\n⚠️  收到中断信号，保存当前模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ RM 完成 → {OUTPUT_DIR}")
