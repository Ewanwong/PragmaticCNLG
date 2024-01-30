from new_PragmaticGPT2 import PragmaticGPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'gpt2'

model = PragmaticGPT2LMHeadModel(model_name, 0, 0, 2).to(device)

prefix = ["You are so beautiful", 'Go to hell', "Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation."]
print(model.classify(prefix, ["It's positive:"], ["It's negative"]))
print(model.debiased_generation(prefix, ["It's positive:"], ["It's negative"], min_length=50, max_length=50, do_sample=True, top_k=5, num_beams=3))

gpt_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt_tokenizer.padding_side = 'left'
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
input_ids = gpt_tokenizer(prefix, return_tensors='pt', padding=True, truncation=True)
input_ids = {k: v.to(device) for k, v in input_ids.items()}
print(gpt_tokenizer.batch_decode(gpt_model.generate(**input_ids, min_length=50, max_length=50, do_sample=True, top_k=5, num_beams=3), skip_special_tokens=True))
