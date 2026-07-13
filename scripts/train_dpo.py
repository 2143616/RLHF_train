"""
RLHF / DPO 训练脚本
使用 TRL 的 DPOTrainer 对 Qwen2.5-1.5B-Instruct 进行偏好对齐训练
"""
import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training

# ===== 配置 =====
MODEL_PATH = "/home/hyl/project/RLHF_train/models/qwen2.5-1.5b"
DATA_PATH = "/home/hyl/project/RLHF_train/data/preference_data.json"
OUTPUT_DIR = "/home/hyl/project/RLHF_train/output/dpo-final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 1. 加载 tokenizer =====
print("=" * 60)
print("Step 1: Loading tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # DPO 需要 left padding

print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
print(f"Pad token: {tokenizer.pad_token}, Pad ID: {tokenizer.pad_token_id}")

# ===== 2. 配置量化 (QLoRA) =====
print("\n" + "=" * 60)
print("Step 2: Loading quantized model...")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 参考模型（与训练模型相同，DPO 需要）
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

print("Models loaded successfully!")

# ===== 3. 配置 LoRA =====
print("\n" + "=" * 60)
print("Step 3: Configuring LoRA...")
print("=" * 60)

lora_config = LoraConfig(
    r=16,                       # LoRA rank
    lora_alpha=32,              # 缩放因子
    lora_dropout=0.05,          # Dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

print(f"LoRA config: rank={lora_config.r}, alpha={lora_config.lora_alpha}")
print(f"Target modules: {lora_config.target_modules}")

# ===== 4. 加载数据 =====
print("\n" + "=" * 60)
print("Step 4: Loading preference data...")
print("=" * 60)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} preference pairs")

# 格式：转换为 Dataset
def format_dpo_data(item):
    """将偏好数据格式化为 DPO 所需格式"""
    # 使用 chat template 格式化 prompt
    prompt_messages = [
        {"role": "system", "content": "你是一个乐于助人、友好且知识渊博的AI助手。你会用中文回答用户的问题，提供详细有用的信息，并拒绝不安全的请求。"},
        {"role": "user", "content": item["prompt"]}
    ]
    
    chosen_messages = [
        {"role": "system", "content": "你是一个乐于助人、友好且知识渊博的AI助手。你会用中文回答用户的问题，提供详细有用的信息，并拒绝不安全的请求。"},
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["chosen"]}
    ]
    
    rejected_messages = [
        {"role": "system", "content": "你是一个乐于助人、友好且知识渊博的AI助手。你会用中文回答用户的问题，提供详细有用的信息，并拒绝不安全的请求。"},
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["rejected"]}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    chosen_text = tokenizer.apply_chat_template(
        chosen_messages, tokenize=False, add_generation_prompt=False
    )
    rejected_text = tokenizer.apply_chat_template(
        rejected_messages, tokenize=False, add_generation_prompt=False
    )
    
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }

# 格式化数据
formatted_data = [format_dpo_data(item) for item in raw_data]

# 拆分为训练集和验证集
train_data = formatted_data[:-5]
eval_data = formatted_data[-5:]

print(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")

train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

print("Sample data (first prompt, truncated):")
print(repr(train_data[0]["prompt"][:200]))

# ===== 5. DPO 训练配置 =====
print("\n" + "=" * 60)
print("Step 5: Setting up DPO training...")
print("=" * 60)

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,      # 等效 batch_size = 8
    learning_rate=5e-5,
    warmup_steps=10,
    num_train_epochs=3,
    logging_steps=5,
    save_steps=50,
    eval_steps=50,
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",                # 8-bit 优化器节省显存
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    dataloader_num_workers=1,
    do_eval=True,
    max_length=1024,
    beta=0.1,                           # KL 惩罚系数
)

print(f"DPO beta (KL penalty): {training_args.beta}")
print(f"Batch size (effective): {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Num epochs: {training_args.num_train_epochs}")

# ===== 6. 创建 DPO Trainer =====
print("\n" + "=" * 60)
print("Step 6: Creating DPOTrainer...")
print("=" * 60)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("DPOTrainer created successfully!")
print(f"Training samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")
print(f"DPO beta (KL penalty): {training_args.beta}")

# ===== 7. 开始训练 =====
print("\n" + "=" * 60)
print("Step 7: Starting DPO training...")
print("=" * 60)

# 预估训练时间
total_steps = (len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
print(f"Total training steps (estimated): {total_steps}")
print(f"Estimated time per step: ~20s (QLoRA on 1.5B model)")
print(f"Estimated total time: ~{total_steps * 20 // 60} minutes")
print()

trainer.train()

print("\n✅ DPO training completed!")

# ===== 8. 保存模型 =====
print("\n" + "=" * 60)
print("Step 8: Saving model...")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 保存训练配置
config_path = os.path.join(OUTPUT_DIR, "training_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL_PATH,
        "data": DATA_PATH,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "learning_rate": training_args.learning_rate,
        "num_epochs": training_args.num_train_epochs,
        "beta": training_args.beta,
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
    }, f, ensure_ascii=False, indent=2)

print(f"✅ Model saved to {OUTPUT_DIR}")
print(f"Training config saved to {config_path}")
print("\n🎉 All done!")
