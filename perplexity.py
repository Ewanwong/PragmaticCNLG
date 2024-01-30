import argparse
import torch
from tqdm import tqdm

from transformers import GPT2TokenizerFast, AutoConfig, AutoTokenizer
from nlp import load_dataset
from PragmaticGPT2 import PragmaticGPT2LMHeadModel

from debiasing import TARGET_PREFIXES, DISTRACTOR_PREFIXES, DISTRACTOR_DEMONSTRATIONS, TARGET_DEMONSTRATIONS

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_filename", type=str, required=True,
                        help="Path to a file to which the output of the self-diagnosis experiment is written")
    parser.add_argument("--model", type=str, default='gpt2-xl',
                        help="The specific model to compute perplexity for (e.g., 'gpt2-medium')")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Value for the rational parameter")
    """
    parser.add_argument("--decay_constant", type=float, default=50,
                        help="Value for the decay constant (lambda in the paper)")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Minimum factor by which each probability is multiplied")
    """
    parser.add_argument("--max_length", type=int, default=-1,
                        help="The maximum input length to be processed (-1 corresponds to the model's context window)")
    parser.add_argument("--max_length_pattern", type=int, default=100,
                        help="The number of tokens to reserve for the self-diagnosis patterns")
    parser.add_argument("--stride", type=int, default=-1,
                        help="If set, for the first --stride tokens no loss is computed")
    parser.add_argument("--use_keywords", action='store_true',
                        help="If set to true, keywords are used instead of full sentences to construct self-debiasing inputs")
    parser.add_argument("--no_cuda", action='store_true',
                        help="If set to true, all computations are done on CPU")
    parser.add_argument("--debug", action='store_true',
                        help="If set, additional debugging output is printed to stdout")
    parser.add_argument("--prior_aggregation_method", type=str, default='sum', choices=['mean', 'sum'],)

    args = parser.parse_args()
    print(f"Parameters: {args}")
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model = PragmaticGPT2LMHeadModel(args.model, args.alpha, 0, len(TARGET_PREFIXES)+len(DISTRACTOR_PREFIXES), args.prior_aggregation_method)
    
    device = 'cuda' if not args.no_cuda else 'cpu'

    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

    max_length = (args.max_length if args.max_length > 0 else config.n_positions) - args.max_length_pattern

    if args.stride <= 0:
        args.stride = max_length

    lls_debiased, lls_regular = [], []
    ppl_debiased, ppl_regular = None, None

    for i in tqdm(range(0, encodings.input_ids.size(1), args.stride)):
        begin_loc = max(i + args.stride - max_length, 0)
        end_loc = min(i + args.stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        input = tokenizer.batch_decode(input_ids)
        
        loss_regular, loss_debiased = model.compute_perplexity(input, TARGET_DEMONSTRATIONS, DISTRACTOR_DEMONSTRATIONS)
        if torch.isnan(loss_debiased).any() or torch.isinf(loss_debiased).any():
            print(f'WARNING: NaN or Inf in debiased loss')
            loss_debiased = torch.stack(lls_debiased).mean() / args.stride
        log_likelihood_debiased = loss_debiased * trg_len
        log_likelihood_regular = loss_regular * trg_len

        lls_debiased.append(log_likelihood_debiased)
        lls_regular.append(log_likelihood_regular)

        ppl_debiased = torch.exp(torch.stack(lls_debiased).sum() / end_loc)
        ppl_regular = torch.exp(torch.stack(lls_regular).sum() / end_loc)
        print(f'Perplexity after {i} tokens: {ppl_debiased} (debiased) vs {ppl_regular} (regular)')

    print(f'Final perplexity: {ppl_debiased} (debiased) vs {ppl_regular} (regular)')

    with open(args.output_filename, 'a', encoding='utf8') as fh:
        fh.write(f'=== RESULT [{args.model}] ===\n')
        fh.write(f'Perplexity (debiased): {ppl_debiased}\n')
        fh.write(f'Perplexity (regular):  {ppl_regular}\n\n')
