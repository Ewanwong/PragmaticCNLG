import json
from new_PragmaticGPT2 import PragmaticGPT2LMHeadModel
from error_analysis_diagnosis import *
from tqdm import tqdm
import torch

examples_file = "w_2_demos_correct.json"
output_file = "w_2_demos_correct_analysis.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(examples_file, 'r') as f:
    examples = json.load(f)

with open(output_file, 'w') as f:
    for model_name in examples.keys():
        model_examples = examples[model_name]
        model = PragmaticGPT2LMHeadModel(model_name, 0, 0, 2).to(device)
        tokenizer = model.tokenizer_right
        for attribute in model_examples.keys():
            f.write(f"================================================================Model: {model_name}, Attribute: {attribute}===========================================================\n")
            attribute_examples = model_examples[attribute]
            target_prompts = target_prompt = fill_in_prefix(TARGET_PREFIXES, DESCRIPTIONS, attribute, 2, True)
            distractor_prompts = fill_in_prefix(DISTRACTOR_PREFIXES, DESCRIPTIONS, attribute, 2, False)
            for example in tqdm(attribute_examples[:50]+attribute_examples[-50:]):
                inputs, labels = model.prepare_target_distractor_inputs([example['text'].replace('\n', '')], target_prompts, distractor_prompts, padding_side='right')
                prior_distributions = model.compute_prior_distributions(**inputs, labels=labels)[:, -(labels[0,:]!=-100).sum():,:]
                f.write(example['text'].replace('\n', ''))
                f.write('\n')
                f.write(f"{model_name}    {attribute}")
                f.write('\n')
                f.write(f"prediction: {example['pred']}")
                f.write('\n')
                f.write(f"actual score: {example['actual']}")
                f.write('\n')
                toks = []
                for tok in inputs['input_ids'][0]:
                    tok = tokenizer.decode(tok)
                    if tok != tokenizer.pad_token:
                        toks.append(tok)
                f.write(' '.join(toks))
                f.write('\n')
                f.write(str(prior_distributions))
                f.write('\n')



