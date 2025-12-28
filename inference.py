import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "nvidia/OpenMath-Nemotron-1.5B"

# 1. Load Model and Tokenizer
# We use BF16 and Flash Attention 2 for the A10G
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

# 2. Prepare the Example from your dataset
# JustRL style prompt with the mandatory suffix
problem = "In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$."
prompt_suffix = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

prompt = problem + prompt_suffix

# Format as a chat message if the model expects it, 
# or use the raw string if testing the base capability.
# Nemotron-1.5B often responds best to the specific format below:
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

# 3. Generate the Reasoning Chain
print("\n--- Generating Solution ---\n")
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=2048, # Enough space for "thinking"
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

# 4. Decode and Print
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
print(response)