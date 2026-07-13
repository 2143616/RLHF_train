"""
测试三阶段模型效果 — SFT / GRPO / DPO
用法: python scripts/test_models.py
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/home/hyl/project/RLHF_train/models/qwen2.5-1.5b"
SFT_PATH = "/home/hyl/project/RLHF_train/output/sft"
GRPO_PATH = "/home/hyl/project/RLHF_train/output/grpo"
DPO_PATH = "/home/hyl/project/RLHF_train/output/dpo-final"

TEST_PROMPTS = [
    "What's the best way to stay healthy?",
    "Explain quantum computing in simple terms.",
    "How can I be more productive at work?",
    "Tell me a short story about a robot learning to love.",
    "What are the pros and cons of remote work?",
]

def load_model(model_path, lora_path=None):
    print(f"  加载: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if lora_path:
        print(f"  合并 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new=128):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    models = [
        ("SFT", SFT_PATH, None),
        ("GRPO", GRPO_PATH, None),
        ("DPO", BASE_MODEL, DPO_PATH),
    ]

    results = {}
    for name, model_path, lora_path in models:
        print(f"\n{'='*60}")
        print(f"  测试 {name} 模型")
        print(f"{'='*60}")
        model, tokenizer = load_model(model_path, lora_path)
        results[name] = []
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            answer = generate(model, tokenizer, prompt)
            results[name].append((prompt, answer))
            print(f"\n  [{i}] {prompt}")
            print(f"  → {answer[:200]}{'...' if len(answer) > 200 else ''}")
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  对比总结")
    print(f"{'='*60}")
    for i in range(len(TEST_PROMPTS)):
        print(f"\n  [{i+1}] {TEST_PROMPTS[i]}")
        for name in ["SFT", "GRPO", "DPO"]:
            ans = next(a for p, a in results[name] if p == TEST_PROMPTS[i])
            print(f"    {name:5s}: {ans[:120]}{'...' if len(ans) > 120 else ''}")
