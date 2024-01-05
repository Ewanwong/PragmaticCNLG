from typing import List, Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, LogitsProcessorList, LogitsProcessor, PreTrainedTokenizer, GPT2Tokenizer
from transformers.generation_utils import GenerationMixin, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions



class StepwiseOutput:
    def __init__(self, unconditional_probabilities, literal_speaker_probabilities, pragmatic_speaker_probabilities, pragmatic_listener_probabilities, prior_probabilities):
        self.unconditional_probabilities = unconditional_probabilities # bsz * len * vocab
        self.literal_speaker_probabilities = literal_speaker_probabilities # bsz * len * vocab
        self.pragmatic_speaker_probabilities = pragmatic_speaker_probabilities # bsz * len * vocab
        self.pragmatic_listener_probabilities = pragmatic_listener_probabilities # # bsz * len * num_classes * vocab
        self.prior_probabilities = prior_probabilities # bsz * len * num_class: first step uniform
        #self.prior_probabilities_next_step = prior_probabilities_next_step  # bsz * num_class * vocab 

class PragmaticGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, model, config, alpha, epsilon, num_classes):
        super().__init__(config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self._device = torch.device("cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model).to(self._device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_classes = num_classes
        print('model initialized')
        

    def prepare_target_distractor_inputs(self, input_texts, target_prompts, distractor_prompts):
        """
        input_texts: a list of strings
        target_prompts: a list of prompts that encourage models to produce attributes of interest
        distractor_prompts: a list of prompts that encourage models to produce attributes as distractors
        """
        inputs = []
        for input_text in input_texts:
        
            inputs.append(input_text)
            for target_prompt in target_prompts:           
                inputs += [target_prompt + input_text]
            for distractor_prompt in distractor_prompts:         
                inputs += [distractor_prompt + input_text]

        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        inputs = {k:v.to(self._device) for k,v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.clone()
        
        # mask out prompts, only calculate the loss for inputs
        input_lengths = attention_mask[[i for i in range(input_ids.shape[0]) if i % (self.num_classes+1) == 0],:].sum(dim=1).cpu()
        
        length_mask = torch.zeros(size=attention_mask.shape)
        for i in range(len(input_texts)):
            length_mask[i*(self.num_classes+1):(i+1)*(self.num_classes+1), -input_lengths[i].item():] = 1
        length_mask = length_mask.to(self._device)
        labels.masked_fill_(length_mask == 0, -100)
        print("inputs prepared")
        
        return inputs, labels
    
    def pragmatic_modeling(self,
        input_ids: Optional[torch.LongTensor] = None, # contains regular+target+distractors
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        
        outputs = self.model(input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # logits, past_key_values, hidden_states, attentions
        logits = outputs.logits # batch, length, vocab
        prob = logits.softmax(dim=-1)
        bsz = logits.shape[0]
        real_bsz = bsz // (self.num_classes+1)
        #print(bsz, real_bsz)
        # calculate prior probabilities
        # print(logits.shape)


        loss_fct = CrossEntropyLoss(reduction='none')
        lm_logits = outputs.logits[[i for i in range(bsz) if i % (self.num_classes+1) != 0], ...]
        if labels is not None:
            # move labels to correct device to enable model parallelism
            prompted_labels = labels[[i for i in range(bsz) if i % (self.num_classes+1) != 0], ...].to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = prompted_labels[..., 1:].contiguous()
            # Flatten the tokens
            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_by_sequence = loss.view(lm_logits.size(0), -1).cumsum(dim=1) # bsz * len
        
        unnormalized_listener_probability = loss_by_sequence.view(real_bsz, self.num_classes, loss_by_sequence.shape[1]).permute(0, 2, 1) # real_bsz * len * num_classes
        
        prior_distributions = F.softmax(unnormalized_listener_probability, dim=-1) # real_bsz * len-1 * num_classes
        prior_distributions = torch.cat(((torch.ones((real_bsz, 1, self.num_classes))/self.num_classes).to(self._device), prior_distributions), dim=1)
        #print(prior_distributions)
        
        
        
        regular_prob = prob[[i for i in range(bsz) if i % (self.num_classes+1) == 0],:,:] # real_bsz * len * vocab
        other_prob = prob[[i for i in range(bsz) if i % (self.num_classes+1) != 0],:,:] # real_bsz x num_classes, len, vocab
     
        
        other_prob_by_example = other_prob.view(real_bsz, self.num_classes, logits.shape[1] , logits.shape[-1]).permute(0, 2, 1, 3) # real_bsz * len * num_classes * vocab
        literal_speaker_probabilities = other_prob_by_example[:, :, 0, :]
        #print(literal_speaker_probabilities.shape)
        pragmatic_listener_probability_distribution = torch.mul(other_prob_by_example, prior_distributions.unsqueeze(-1).expand(-1, -1, -1, logits.shape[-1])) # real_bsz * len * num_classes * vocab
        pragmatic_listener_probability_distribution = F.normalize(pragmatic_listener_probability_distribution, p=1.0, dim=2) # real_bsz * len * num_classes * vocab -> prior for next step depending on what token is selected
        #print(pragmatic_listener_probability_distribution.shape)
        
        evidence = torch.sum(torch.mul(pragmatic_listener_probability_distribution[:,:,0,:] ** self.alpha, regular_prob), dim=[0, 1])
        #print(evidence.shape)
        mask = (pragmatic_listener_probability_distribution[:, :, 0, :] ** self.alpha) / evidence
        #print(mask)
        mask = torch.max(mask, torch.tensor([self.epsilon], device=mask.device))
        pragmatic_speaker_probability_distribution = torch.mul(regular_prob, mask)
        pragmatic_speaker_probability_distribution = F.normalize(pragmatic_speaker_probability_distribution, p=1.0, dim=2) # real_bsz * len * vocab
        #print(pragmatic_speaker_probability_distribution.shape)
        
        loss = None
        lm_logits = torch.log(pragmatic_speaker_probability_distribution)
        if labels is not None:
            prompted_labels = labels[[i for i in range(bsz) if i % (self.num_classes+1) == 0], :].to(lm_logits.device)
            # move labels to correct device to enable model parallelism
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = prompted_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity_loss = loss
        

        return outputs, StepwiseOutput(regular_prob, literal_speaker_probabilities, pragmatic_speaker_probability_distribution, pragmatic_listener_probability_distribution, prior_distributions), perplexity_loss
    
    def classify(self, input_texts, target_prompts, distractor_prompts):
        inputs, labels = self.prepare_target_distractor_inputs(input_texts, target_prompts, distractor_prompts)
        outputs, step_output, perplexity_loss = self.pragmatic_modeling(**inputs, labels=labels)
        pred = step_output.prior_probabilities[:, -1, :]
        return pred
    


    def sample():
        pass










