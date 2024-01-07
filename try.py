from PragmaticGPT2 import PragmaticGPT2LMHeadModel
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel
import torch
model_name = 'gpt2-xl'
config = AutoConfig.from_pretrained(model_name)
model = PragmaticGPT2LMHeadModel(model_name, config, 1, 0, 2)

target_prompt = ["The following sentence is non-toxic: "]
distractor_prompts = ["The following sentence is toxic: "]
                      

prompts = ["I don't like her because she",
           "When I spend time with her, I feel",
           "The additional split extra textual administration?"]
#print(model.classify(prompts, target_prompt, distractor_prompts))

print(model.debiased_generation(prompts, target_prompt, distractor_prompts, min_length=0, max_length=40, do_sample=True, top_k=5, use_cache=False))


gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normal_generation(model, tokenizer, device, prompts):

    model.to(device)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
    inputs = {k:v.to(device) for k,v in inputs.items()}
    return tokenizer.batch_decode(model.generate(**inputs, min_length=0, max_length=40, do_sample=True, top_k=5), skip_special_tokens=True)

print(normal_generation(gpt_model, tokenizer, device, prompts))
