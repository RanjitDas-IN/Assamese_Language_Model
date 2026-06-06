from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
import torch

ckpt = "/home/ranjit/Desktop/projects/Laguage_Model/hf_download/checkpoint-19000"

tokenizer = AutoTokenizer.from_pretrained(ckpt)

model = AutoModelForCausalLM.from_pretrained(
    ckpt,
    dtype=torch.float16,
    device_map="auto",
)

prompt = "এদিন এখন সৰু গাঁৱত "
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

generation_kwargs = dict(
    **inputs,
    streamer=streamer,
    max_new_tokens=512,  # or 1024
    do_sample=True,
    temperature=1.0,
    top_p=0.75,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

print(prompt, end="", flush=True)

for token in streamer:
    print(token, end="", flush=True)

print()