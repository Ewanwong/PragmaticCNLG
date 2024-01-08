from typing import List, Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, LogitsProcessorList, LogitsProcessor, PreTrainedTokenizer, GPT2Tokenizer
from transformers.generation_utils import GenerationMixin, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput, GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput, BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer, BeamHypotheses
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

        self.model.eval()
        torch.set_grad_enabled(False)
        
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
        inputs += input_texts
        
        for target_prompt in target_prompts:  
            for input_text in input_texts:         
                inputs += [target_prompt + input_text]
        for distractor_prompt in distractor_prompts:
            for input_text in input_texts:         
                inputs += [distractor_prompt + input_text]

        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        inputs = {k:v.to(self._device) for k,v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.clone()
        
        # mask out prompts, only calculate the loss for inputs
        input_lengths = attention_mask[:input_ids.shape[0]//(self.num_classes+1),:].sum(dim=1).cpu()
        
        length_mask = torch.zeros(size=attention_mask.shape)
        for i in range(len(input_texts)):
            length_mask[[i+k*(len(input_texts)) for k in range(self.num_classes+1)], -input_lengths[i].item():] = 1
        length_mask = length_mask.to(self._device)
        labels.masked_fill_(length_mask == 0, -100)
        #print("inputs prepared")
        
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
        lm_logits = outputs.logits[real_bsz:, ...]
        if labels is not None:
            # move labels to correct device to enable model parallelism
            prompted_labels = labels[real_bsz:, ...].to(lm_logits.device)
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
        
        
        
        regular_prob = prob[:real_bsz,:,:] # real_bsz * len * vocab
        other_prob = prob[real_bsz:,:,:] # real_bsz x num_classes, len, vocab
     
        
        other_prob_by_example = other_prob.view(self.num_classes, real_bsz, logits.shape[1] , logits.shape[-1]).permute(1, 2, 0, 3) # real_bsz * len * num_classes * vocab
        literal_speaker_probabilities = other_prob_by_example[:, :, 0, :]
        #print(literal_speaker_probabilities.shape)
        pragmatic_listener_probability_distribution = torch.mul(other_prob_by_example, prior_distributions.unsqueeze(-1).expand(-1, -1, -1, logits.shape[-1])) # real_bsz * len * num_classes * vocab
        pragmatic_listener_probability_distribution = pragmatic_listener_probability_distribution / pragmatic_listener_probability_distribution.sum(dim=2).unsqueeze(dim=2) # real_bsz * len * num_classes * vocab -> prior for next step depending on what token is selected
        #print(pragmatic_listener_probability_distribution.shape)
        """
        evidence = torch.sum(torch.mul(pragmatic_listener_probability_distribution[:,:,0,:] ** self.alpha, regular_prob), dim=[1, 2]).view(-1, 1, 1)
        
        mask = (pragmatic_listener_probability_distribution[:, :, 0, :] ** self.alpha) / evidence
        
        mask = torch.max(mask, torch.tensor([self.epsilon], device=mask.device))
        pragmatic_speaker_probability_distribution = torch.mul(regular_prob, mask)
        """
        pragmatic_speaker_probability_distribution = torch.mul(pragmatic_listener_probability_distribution[:,:,0,:] ** self.alpha, regular_prob)
        pragmatic_speaker_probability_distribution = pragmatic_speaker_probability_distribution / pragmatic_speaker_probability_distribution.sum(dim=-1).unsqueeze(dim=-1) # real_bsz * len * vocab
        
        #print(((pragmatic_speaker_probability_distribution-regular_prob)**2).sum()) # measure difference between regular and debiased distributions
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
        output_ids = self.generate(**inputs, labels=labels, min_length=min_length, max_length=max_length, **kwargs)
        if output_ids.shape[0] == inputs['input_ids'].shape[0]: # beam search returns real batch size
            output_ids = output_ids[:(output_ids.shape[0]//(self.num_classes+1)), ...]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    # rewrite sample function from GenerationMixin
    def sample(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
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
            #print("*********************************")
            # prepare model inputs
           
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            #print(f"input_ids shape:{model_inputs['input_ids'].shape}")
        ################################################################################################
        # use pragmatic output for generation
            # forward pass to get next token
            outputs, stepwise_outputs, perplexity_loss = self.pragmatic_modeling(
                **model_inputs,
                labels=labels,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            #next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = torch.log(stepwise_outputs.pragmatic_speaker_probabilities[:, -1, :])

            # pre-process distribution
            real_input_ids = input_ids[:(input_ids.shape[0]//(self.num_classes+1)), ...]
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
            #################################################################################################
            #duplicate next tokens before adding to existing tokens
            next_tokens = torch.stack([next_tokens] * (self.num_classes+1)).reshape(-1,)
            #################################################################################################
            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            #################################################################################################
            
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1
            labels = torch.cat([labels, input_ids[:, -1:]], dim=-1)
            #print(f"labels shape: {labels.shape}")
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


    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        """
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        """
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            """
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
            """    
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
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

        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs, stepwise_outputs, perplexity_loss = self.pragmatic_modeling(
                **model_inputs,
                labels=labels,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            #next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = torch.log(stepwise_outputs.pragmatic_speaker_probabilities[:, -1, :])

            
            

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_logits,)
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

            # pre-process distribution
            
            real_input_ids = input_ids[:(input_ids.shape[0]//(self.num_classes+1)), ...]
            next_tokens_scores = logits_processor(real_input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            next_tokens = torch.stack([next_tokens] * (self.num_classes+1)).reshape(-1,)
            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            labels = torch.cat([labels, input_ids[:, -1:]], dim=-1)

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # increase cur_len
            cur_len = cur_len + 1

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids


    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        labels: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
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
        ########################################################################

        num_beams = beam_scorer.num_beams
        labels = labels.repeat(num_beams, 1)
        #print(input_ids.shape)
        batch_size = input_ids.shape[0] // ((self.num_classes+1) * num_beams)

        #batch_size = len(beam_scorer._beam_hyps)
        beam_scorer.batch_size = batch_size
        beam_scorer._beam_hyps = [
            BeamHypotheses(
                num_beams=beam_scorer.num_beams,
                max_length=beam_scorer.max_length,
                length_penalty=beam_scorer.length_penalty,
                early_stopping=beam_scorer.do_early_stopping,
            )
            for _ in range(beam_scorer.batch_size)
        ]
        beam_scorer._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=beam_scorer.device)
        ########################################################################
        

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == (batch_beam_size // (self.num_classes+1))
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs, stepwise_outputs, perplexity_loss = self.pragmatic_modeling(
                **model_inputs,
                labels=labels,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            #print(((stepwise_outputs.pragmatic_speaker_probabilities[:, -1, :]-F.softmax(outputs.logits[:(outputs.logits.shape[0]//(self.num_classes+1)), -1, :], dim=-1))**2).sum())
            next_token_logits = torch.log(stepwise_outputs.pragmatic_speaker_probabilities[:, -1, :])

            # adjust tokens for Bart, *e.g.*
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            ################################################################
            real_input_ids = input_ids[:(input_ids.shape[0]//(self.num_classes+1)), ...]
            next_token_scores = logits_processor(real_input_ids, next_token_scores)
            ################################################################
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

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

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            ##################################################################
            beam_outputs = beam_scorer.process(
                real_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            ########################################################################
            
            #print(beam_idx)
            
            beam_next_tokens = torch.cat([beam_next_tokens]*(self.num_classes+1)).reshape(-1)
            beam_idx = torch.cat([beam_idx]*(self.num_classes+1)).reshape(-1)
            
            beam_idx = beam_idx+torch.tensor([(i//(num_beams*(self.num_classes+1)))*num_beams*self.num_classes for i in range(beam_idx.shape[0])], device=beam_idx.device)
            #print(beam_idx)
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            
            cur_len = cur_len + 1
            labels = torch.cat([labels, input_ids[:,labels.shape[1]:]], dim=-1)
            ########################################################################
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break
            #print(input_ids[[i for i in range(input_ids.shape[0]) if (i//num_beams) % (self.num_classes+1) == 0], ...])
            
            """
            print(f"input_ids shape: {input_ids.shape}")
            print(f"labels shape: {labels.shape}")
            print(f"beam_scores shape: {beam_scores.shape}")
            print(f"beam_next_tokens shape: {beam_next_tokens.shape}")
            print(f"beam_idx shape: {beam_idx.shape}")
            print(f"next_token_scores shape: {next_token_scores.shape}")
            print(f"next_tokens shape: {next_tokens.shape}")
            print(f"next_indices shape: {next_indices.shape}")
            """
            #print(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True))
        
        sequence_outputs = beam_scorer.finalize(
            input_ids[:(input_ids.shape[0]//(self.num_classes+1)), :], beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
    
    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        labels: torch.LongTensor,
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
    ) -> Union[BeamSampleOutput, torch.LongTensor]:

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
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

        ########################################################################

        num_beams = beam_scorer.num_beams
        labels = labels.repeat(num_beams, 1)
        #print(input_ids.shape)
        batch_size = input_ids.shape[0] // ((self.num_classes+1) * num_beams)

        #batch_size = len(beam_scorer._beam_hyps)
        beam_scorer.batch_size = batch_size
        beam_scorer._beam_hyps = [
            BeamHypotheses(
                num_beams=beam_scorer.num_beams,
                max_length=beam_scorer.max_length,
                length_penalty=beam_scorer.length_penalty,
                early_stopping=beam_scorer.do_early_stopping,
            )
            for _ in range(beam_scorer.batch_size)
        ]
        beam_scorer._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=beam_scorer.device)
        ########################################################################

        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs, stepwise_outputs, perplexity_loss = self.pragmatic_modeling(
                **model_inputs,
                labels=labels,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = torch.log(stepwise_outputs.pragmatic_speaker_probabilities[:, -1, :])

            # adjust token scores (a no-op by default)
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            real_input_ids = input_ids[:(input_ids.shape[0]//(self.num_classes+1)), ...]
            next_token_scores = logits_processor(real_input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(real_input_ids, next_token_scores)

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

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                real_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            ########################################################################
            beam_next_tokens = torch.cat([beam_next_tokens]*(self.num_classes+1)).reshape(-1)
            beam_idx = torch.cat([beam_idx]*(self.num_classes+1)).reshape(-1)
            
            beam_idx = beam_idx+torch.tensor([(i//(num_beams*(self.num_classes+1)))*num_beams*self.num_classes for i in range(beam_idx.shape[0])], device=beam_idx.device)
            #print(beam_idx)
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            
            cur_len = cur_len + 1
            labels = torch.cat([labels, input_ids[:,labels.shape[1]:]], dim=-1)
            ########################################################################

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids[:(input_ids.shape[0]//(self.num_classes+1)), :], beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]