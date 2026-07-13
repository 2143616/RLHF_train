"""
准备完整的 RLHF 三阶段数据集（SFT + RM + PPO）
数据源：Anthropic/hh-rlhf（从 hf-mirror 加载）
"""
import os, json, re

DATA_DIR = "/home/hyl/project/RLHF_train/data"
os.makedirs(DATA_DIR, exist_ok=True)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset

ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=False)
print(f"总数据量: {len(ds)} 条")

# ===== 2. 解析 Human/Assistant 对话格式 =====
print("\nStep 2: 解析对话数据...")

def parse_dialog(text):
    """从 Human/Assistant 对话中提取 prompt 和 response"""
    # 格式: "\n\nHuman: xxx\n\nAssistant: xxx"
    parts = text.split("\n\nAssistant: ")
    if len(parts) < 2:
        return None, None
    prompt_part = parts[0]
    response = parts[1].strip()
    # 提取 Human 部分
    human_match = re.search(r"Human: (.*)", prompt_part, re.DOTALL)
    if not human_match:
        return None, None
    prompt = human_match.group(1).strip()
    return prompt, response

# ===== 3. 构建 SFT 数据集 =====
print("\nStep 3: 构建 SFT 数据（取 chosen 中的 Human/Assistant 对）...")

sft_data = []
for i, row in enumerate(ds):
    prompt, response = parse_dialog(row["chosen"])
    if prompt and response and 10 < len(prompt) < 1000 and 10 < len(response) < 2000:
        sft_data.append({
            "instruction": prompt,
            "output": response
        })

print(f"SFT 数据: {len(sft_data)} 条")

# 保存 SFT 数据
sft_path = f"{DATA_DIR}/sft_data.json"
with open(sft_path, "w", encoding="utf-8") as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=2)
print(f"已保存: {sft_path}")

# ===== 4. 构建 RM 数据集（偏好对） =====
print("\nStep 4: 构建 RM 偏好数据...")

rm_data = []
for i, row in enumerate(ds):
    prompt_c, resp_c = parse_dialog(row["chosen"])
    prompt_r, resp_r = parse_dialog(row["rejected"])
    if prompt_c and resp_c and prompt_r and resp_r and prompt_c == prompt_r:
        rm_data.append({
            "prompt": prompt_c,
            "chosen": resp_c,
            "rejected": resp_r
        })

print(f"RM 数据: {len(rm_data)} 条")

# 保存 RM 数据
rm_path = f"{DATA_DIR}/rm_data.json"
with open(rm_path, "w", encoding="utf-8") as f:
    json.dump(rm_data, f, ensure_ascii=False, indent=2)
print(f"已保存: {rm_path}")

# ===== 5. 构建 PPO 使用的 prompts =====
print("\nStep 5: 构建 PPO prompts...")

ppo_prompts = []
for item in rm_data[:50]:  # 取前 50 条 prompt
    ppo_prompts.append({"prompt": item["prompt"]})

ppo_path = f"{DATA_DIR}/ppo_prompts.json"
with open(ppo_path, "w", encoding="utf-8") as f:
    json.dump(ppo_prompts, f, ensure_ascii=False, indent=2)
print(f"PPO prompts: {len(ppo_prompts)} 条")
print(f"已保存: {ppo_path}")

# ===== 6. 汇总 =====
print("\n" + "=" * 55)
print("数据集准备完毕！")
print("=" * 55)
print(f"  SFT (指令微调):  {len(sft_data):>4} 条 → data/sft_data.json")
print(f"  RM  (奖励模型):  {len(rm_data):>4} 条 → data/rm_data.json")
print(f"  PPO (prompts):   {len(ppo_prompts):>4} 条 → data/ppo_prompts.json")
print(f"  数据总量:         {len(sft_data) + len(rm_data) + len(ppo_prompts):>4} 条")
print(f"  数据来源:         Anthropic/hh-rlhf (从 hf-mirror 加载)")
print("=" * 55)

# 展示示例
print("\n📌 数据示例:")
print(f"\n  [SFT] 指令:  {sft_data[0]['instruction'][:80]}...")
print(f"  [SFT] 回答:  {sft_data[0]['output'][:80]}...")
print(f"\n  [RM] prompt:   {rm_data[0]['prompt'][:80]}...")
print(f"  [RM] chosen:   {rm_data[0]['chosen'][:80]}...")
print(f"  [RM] rejected: {rm_data[0]['rejected'][:80]}...")
