from PragmaticGPT2 import PragmaticGPT2LMHeadModel
from transformers import AutoConfig, AutoTokenizer
import torch
model_name = 'gpt2-xl'
config = AutoConfig.from_pretrained(model_name)
model = PragmaticGPT2LMHeadModel(model_name, config, 1, 1e-2, 2)

target_prompt = ["The following sentence is non-toxic: "]
distractor_prompts = ["The following sentence is toxic: "]
                      

prompts = ["I don't like her because she is a bitch.",
           "When I spend time with her, I feel very annoyed.",
           "The additional split extra textual administration?"]

print(model.classify(prompts, target_prompt, distractor_prompts))