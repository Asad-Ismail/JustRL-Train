import re, os, torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

os.environ["WANDB_PROJECT"] = "JustRL-Nemotron"

model_id = "nvidia/OpenMath-Nemotron-1.5B"
PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

# Dataset Prep with Truncation (Handles the warning)
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", split="train")

# Load Evaluation Data
aime_eval = load_dataset("HuggingFaceH4/aime_2024", split="train")

def format(ex):
    if "prompt" in ex:
        prompt_text = ex["prompt"]
    elif "problem" in ex:
        prompt_text = ex["problem"]
    else:
        raise ValueError(f"No prompt field in example keys: {ex.keys()}")

    prompt_text = prompt_text + PROMPT_SUFFIX

    if "solution" in ex:
        solution = ex["solution"]
    elif "answer" in ex:
        solution = ex["answer"]
    else:
        raise ValueError(f"No answer field in example keys: {ex.keys()}")

    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "answer": str(solution).strip(),
    }



# Prepare Datasets
train_dataset = dataset.map(format)
eval_dataset = aime_eval.map(format)


def math_reward(completions, answer, **kwargs):
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        matches = re.findall(r"\\boxed\{(.*?)\}", content)
        rewards.append(1.0 if (matches and matches[-1].strip() == ground_truth) else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    """Small reward for just using the \boxed{} tag. Breaks the 0-gradient loop."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        # Reward 0.2 if the tag exists, 0.0 otherwise
        rewards.append(0.2 if "\\boxed{" in content else 0.0)
    return rewards

def brevity_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        # Penalize if reasoning is too long, reward if concise
        content = completion[0]["content"]
        token_count = len(content.split())
        if token_count > 8000:
            rewards.append(-0.5) # Heavy penalty for hitting the "clipping danger zone"
        else:
            rewards.append(0.0)
    return rewards

# GRPO Config 
training_args = GRPOConfig(
    output_dir="./justrl-original-weights",
    learning_rate=1e-6,
    lr_scheduler_type="constant",
    max_steps=3440,                # JustRL Paper Step count
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, # (1 * 8 * 8 *4) = 256 Global Batch Size
    num_generations=8,             # Keep at 8 for A10G VRAM safety
    max_completion_length=13000,     # Safer for 24GB VRAM
    beta=0.0,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # vLLM Acceleration
    use_vllm=True,
    vllm_mode="colocate",               # Embeds vLLM in each training process
    vllm_gpu_memory_utilization=0.17,    # Reserve 20% for generation rollouts
    vllm_max_model_length=14000,
    deepspeed="./configs/ds_config_3.json",
    model_init_kwargs={
        "dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
         "device_map": None,
    },
    report_to="wandb",
    logging_steps=1,
    eval_strategy="steps",           # Evaluate periodically
    eval_steps=50,                  # Run evaluation every 50 steps
    per_device_eval_batch_size=1,    # Keep low for VRAM safety
    num_generations_eval=4,

    save_strategy="steps",          # Must match eval_strategy
    save_steps=50,                  # Save at the same frequency as eval
    save_total_limit=2,             # Keep only the best and the most recent checkpoint
    load_best_model_at_end=True,    # Automatically load the best model when training finishes
    
    # This must match the name of your math reward function
    metric_for_best_model="eval/rewards/math_reward/mean", 
    greater_is_better=True,   
)

trainer = GRPOTrainer(
    model=model_id,
    reward_funcs=[math_reward,format_reward,brevity_reward],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()