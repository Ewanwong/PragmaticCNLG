import argparse
import torch
import json
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from os import listdir
from os.path import isfile, join


def load_data(examples_filename):
    with open(examples_filename, 'r') as f:
        examples = f.readlines()
    examples = [json.loads(example)['continuations'][0]['text'] for example in examples]
    return examples

def compute_perplexity(examples, model, tokenizer, device):
    perplexities = []
    for example in tqdm(examples):
        input_ids = tokenizer.encode(example, return_tensors='pt').to(device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexities.append(loss.item()*(input_ids.shape[1]-1))
    return perplexities

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples_dir", type=str, required=True,
                        help="Path to a file to which the output of the self-diagnosis experiment is written")
    
    parser.add_argument("--perplexity_model", type=str, default='EleutherAI/gpt-j-6b', # EleutherAI/gpt-j-6b
                        help="The specific model to compute perplexity for (e.g., 'gpt2-medium')")

    args = parser.parse_args()
    print(f"Parameters: {args}")

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.perplexity_model)
    model = AutoModelForCausalLM.from_pretrained(args.perplexity_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    torch.set_grad_enabled(False)
    model.to(device)

    # find all txt files in a directory
    example_files = [f for f in listdir(args.examples_dir) if isfile(join(args.examples_dir, f)) and f.endswith('.txt') and 'prompted_generations_' in f]
    output_files = [f.replace('prompted_generations_', 'perplexity_') for f in example_files]

    for examples_file, output_file in zip(example_files, output_files):
        print(examples_file)
        # load data from file
        examples = load_data(join(args.examples_dir, examples_file))



        # compute perplexity
        perplexities = compute_perplexity(examples, model, tokenizer, device)

        # write output to file
        with open(join(args.examples_dir, output_file), 'w') as f:
            
            f.write('Average perplexity of model output: ' + str(sum(perplexities)/len(perplexities)) + '\n')