from typing import List, Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, LogitsProcessorList, LogitsProcessor, PreTrainedTokenizer, GPT2Tokenizer
from transformers.generation_utils import GenerationMixin, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput, GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput, BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]

class StepwiseOutput:
    def __init__(self, unconditional_probabilities, literal_speaker_probabilities, pragmatic_speaker_probabilities, pragmatic_listener_probabilities, prior_probabilities):
        self.unconditional_probabilities = unconditional_probabilities # bsz * len * vocab
        self.literal_speaker_probabilities = literal_speaker_probabilities # bsz * len * vocab
        self.pragmatic_speaker_probabilities = pragmatic_speaker_probabilities # bsz * len * vocab
        self.pragmatic_listener_probabilities = pragmatic_listener_probabilities # # bsz * len * num_classes * vocab
        self.prior_probabilities = prior_probabilities # bsz * len * num_class: first step uniform
        #self.prior_probabilities_next_step = prior_probabilities_next_step  # bsz * num_class * vocab 

class PragmaticGPT2LMHeadModel(GPT2LMHeadModel, GenerationMixin):
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
    
    def debiased_generation(self, input_texts, target_prompts, distractor_prompts, min_length: int = None, max_length: int = None, **kwargs):
        inputs, labels = self.prepare_target_distractor_inputs(input_texts, target_prompts, distractor_prompts)
        input_length = inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length
        output_ids = self.generate(**inputs, min_length=min_length, max_length=max_length, **kwargs)
        output_ids = output_ids[[i for i in range(output_ids.shape[0]) if i % (self.num_classes+1) == 0], ...]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    # rewrite sample function from GenerationMixin
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using multinomial sampling.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            logits_warper (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsWarper` used to warp the prediction score distribution of the language
                modeling head applied before multinomial sampling at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.SampleEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.SampleDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.SampleEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForCausalLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    TopKLogitsWarper,
            ...    TemperatureLogitsWarper,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])
            >>> # instantiate logits processors
            >>> logits_warper = LogitsProcessorList([
            ...     TopKLogitsWarper(50),
            ...     TemperatureLogitsWarper(0.7),
            ... ])

            >>> outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )

        # auto-regressive generation
        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        ################################################################################################
        # use pragmatic output for generation
            # forward pass to get next token
            outputs, stepwise_outputs, perplexity_loss = self.pragmatic_modeling(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            #next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = stepwise_outputs.pragmatic_speaker_probabilities[:, -1, :]

            # pre-process distribution
            real_input_ids = input_ids[[i for i in range(input_ids.shape[0]) if i % (self.num_classes+1) == 0], ...]
            next_token_scores = logits_processor(real_input_ids, next_token_logits)
            next_token_scores = logits_warper(real_input_ids, next_token_scores)
        #################################################################################################
            
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            #################################################################################################
            #duplicate next tokens before adding to existing tokens
            next_tokens = torch.stack([next_tokens] * (self.num_classes+1)).permute(1, 0).reshape(-1, 0)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1
            #################################################################################################

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids









