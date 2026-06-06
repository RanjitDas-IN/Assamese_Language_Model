import torch
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

MODEL_PATH = "/kaggle/working/outputs"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# Prompt
prompt = "অসমৰ বিষয়ে চমুকৈ কওক।"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Streamer
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

# Generation settings
generation_kwargs = {
    **inputs,
    "streamer": streamer,
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# Generate in background
thread = Thread(
    target=model.generate,
    kwargs=generation_kwargs,
)
thread.start()

# Stream output
print("Assistant: ", end="", flush=True)
for new_text in streamer:
    print(new_text, end="", flush=True)

print()