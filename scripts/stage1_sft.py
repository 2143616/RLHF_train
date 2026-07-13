"""
Stage 1: SFT (Supervised Fine-Tuning) — 全参数微调
适配完整 159K 数据集，使用 map 分批 tokenize
"""
import os, json, torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, DataCollatorForSeq2Seq,
)
from transformers import Trainer as HFTrainer
from datasets import load_dataset

MODEL_PATH = "/home/hyl/project/RLHF_train/models/qwen2.5-1.5b"
DATA_PATH = "/home/hyl/project/RLHF_train/data/sft_data.json"
OUTPUT_DIR = "/home/hyl/project/RLHF_train/output/sft"

MAX_STEPS = int(os.environ.get("SFT_STEPS", "0"))
NUM_EPOCHS = float(os.environ.get("SFT_EPOCHS", "1"))

# 步数模式 vs epoch 模式
if MAX_STEPS > 0:
    HF_MAX_STEPS = MAX_STEPS
    HF_NUM_EPOCHS = 0.0  # Trainer 要求数值比较
else:
    HF_MAX_STEPS = -1
    HF_NUM_EPOCHS = NUM_EPOCHS
VALIDATION_SPLIT = min(int(os.environ.get("SFT_VALID", "5000")), 10000)

os.makedirs(OUTPUT_DIR, exist_ok=True)

mode = f"{MAX_STEPS} 步" if MAX_STEPS else f"{NUM_EPOCHS} epoch"
print(f"=" * 55)
print(f"Stage 1: SFT 全参数 | {mode}")
print(f"=" * 55)

# 1. Tokenizer
print("[1/4] 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. 模型
print("[2/4] 加载模型（全参数 bf16）...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
print(f"  参数量: {model.num_parameters():,} (全部可训练)")

# 3. 数据 — 分批 tokenize
print("[3/4] 加载数据...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train", streaming=False)
print(f"  原始: {len(dataset)} 条")

def tokenize_fn(examples):
    texts = []
    for inst, out in zip(examples["instruction"], examples["output"]):
        texts.append(tokenizer.apply_chat_template([
            {"role": "user", "content": inst},
            {"role": "assistant", "content": out},
        ], tokenize=False))
    enc = tokenizer(texts, truncation=True, max_length=512, padding=False)
    enc["labels"] = enc["input_ids"][:]
    return enc

dataset = dataset.map(tokenize_fn, batched=True, batch_size=1000,
                       remove_columns=["instruction", "output"])
dataset = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=42)
print(f"  训练: {len(dataset['train'])} 条, 验证: {len(dataset['test'])} 条")

# 4. 训练
print("[4/4] 配置 Trainer...")
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_steps=HF_MAX_STEPS,
    num_train_epochs=HF_NUM_EPOCHS,
    logging_steps=10,
    save_steps=500, save_total_limit=2,
    eval_strategy="steps", eval_steps=500,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",
    lr_scheduler_type="cosine", warmup_steps=100,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=2,
    ddp_find_unused_parameters=False,
)

trainer = HFTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

print(f"\n开始训练...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ SFT 完成 → {OUTPUT_DIR}")
