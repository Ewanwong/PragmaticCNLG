import json
from typing import List, Optional, Dict, Any
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ModelOutput:
    """This class represents a piece of text generated by a language model, as well as corresponding attribute scores"""

    TEXT_REPR_MAX_LEN = 50

    def __init__(self, text: str, scores: Dict[str, float]):
        """
        :param text: the generated text
        :param scores: the attribute scores
        """
        self.text = text
        self.scores = scores

    def __repr__(self) -> str:
        text_shortcut = self.text.replace('\n', ' ')
        if len(text_shortcut) > ModelOutput.TEXT_REPR_MAX_LEN:
            text_shortcut = text_shortcut[:ModelOutput.TEXT_REPR_MAX_LEN] + '...'
        return f'Example(text="{text_shortcut}", scores={self.scores})'

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of this output"""
        return {'text': self.text, 'scores': self.scores}


class Prompt:
    """This class represents a prompt, optionally with a list of corresponding model outputs"""

    def __init__(self, text: str, challenging: bool, continuations: Optional[List[ModelOutput]] = None):
        """
        :param text: the actual prompt text
        :param challenging: a flag indicating whether this prompt is considered challenging
        :param continuations: an optional list of continuations
        """
        self.text = text
        self.challenging = challenging
        self.continuations = continuations if continuations is not None else []

    def __repr__(self) -> str:
        return f'Prompt(text="{self.text}", challenging={self.challenging}, continuations=[{len(self.continuations)} Continuations])'

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of this prompt"""
        return {'prompt': self.text, 'challenging': self.challenging, 'continuations': [cont.to_dict() for cont in self.continuations]}


def load_model_outputs(filename: str) -> List[ModelOutput]:
    """Load model outputs from a jsonl file in the RealToxicityPrompts format"""
    print(f'Loading model outputs from file "{filename}"')
    examples = []
    with open(filename, 'r', encoding='utf8') as fh:
        for line in fh:
            line_json = json.loads(line)

            if "generations" in line_json:
                line_examples = line_json["generations"]
            else:
                line_examples = [line_json]

            for ex_json in line_examples:
                text = ex_json['text']
                del ex_json['text']
                example = ModelOutput(text=text, scores=ex_json)
                examples.append(example)

    filtered_examples = [example for example in examples if not any(score is None for score in example.scores.values())]
    print(f'Done loading {len(filtered_examples)} ({len(examples)} before filtering) examples from file "{filename}"')
    return filtered_examples


def load_prompts(filename: str, challenging_only: bool = False) -> List[Prompt]:
    """Load prompts from a jsonl file in the RealToxicityPrompts format"""
    print(f'Loading prompts from file "{filename}"')
    prompts = []
    with open(filename, 'r', encoding='utf8') as fh:
        for line in fh:
            line_json = json.loads(line)
            prompt = Prompt(text=line_json['prompt']['text'], challenging=line_json['challenging'])
            if not challenging_only or prompt.challenging:
                prompts.append(prompt)
    print(f'Done loading {len(prompts)} {"challenging " if challenging_only else ""}prompts from file "{filename}"')
    return prompts
