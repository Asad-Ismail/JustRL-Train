import re, os, torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

os.environ["WANDB_PROJECT"] = "JustRL-Nemotron-Original"

model_id = "nvidia/OpenMath-Nemotron-1.5B"
PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

# Dataset Prep with Truncation (Handles the warning)
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", split="train")

def format_and_truncate(ex):
    prompt_text = ex["prompt"] + PROMPT_SUFFIX
    # Simple character-based truncation for safety; ideally use tokenizer.encode
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "answer": str(ex["solution"]).strip()
    }
dataset = dataset.map(format_and_truncate)

#  Reward Logic
def justrl_reward_func(completions, answer, **kwargs):
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        matches = re.findall(r"\\boxed\{(.*?)\}", content)
        rewards.append(1.0 if (matches and matches[-1].strip() == ground_truth) else 0.0)
    return rewards

# GRPO Config 
training_args = GRPOConfig(
    output_dir="./justrl-original-weights",
    learning_rate=1e-6,
    lr_scheduler_type="constant",
    max_steps=3440,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # Increased for 1-GPU Global Batch 256
    num_generations=8,              
    beta=0.0,
    bf16=True,
    gradient_checkpointing=True,
    model_init_kwargs={
        "dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
         "device_map": None,
    },
    report_to="wandb",
    save_steps=500,
    logging_steps=1,
)

#  Initialize Trainer (No model_init_kwargs here)
trainer = GRPOTrainer(
    model=model_id,
    reward_funcs=[justrl_reward_func],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()