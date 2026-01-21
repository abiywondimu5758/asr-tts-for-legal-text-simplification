# test_mt5_amharic_qa.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_id = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

text = (
    "answer: "
    "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት። "
    "ጥያቄ፡ አዲስ አበባ ምንድን ናት?"
)

inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
