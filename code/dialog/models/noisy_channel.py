import copy
import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, BartModel, PretrainedBartModel, PretrainedConfig, PreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.generation_utils import BeamSearchEncoderDecoderOutput, GenerationMixin, BeamSearchOutput, BeamSampleOutput
from transformers.generation_stopping_criteria import validate_stopping_criteria
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import LogitsProcessorList, LogitsWarper
from transformers.generation_stopping_criteria import StoppingCriteriaList


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class CustomModelOutput(ModelOutput):
    last_hidden_states = None
    
    
class TypicalLogitsWarper(LogitsWarper):
     def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):

         self.filter_value = filter_value
         self.mass = mass
         self.min_tokens_to_keep = min_tokens_to_keep

     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

         # calculate entropy
         normalized = torch.nn.functional.log_softmax(scores, dim=-1)
         p = torch.exp(normalized)
         ent = -(normalized * p).nansum(-1, keepdim=True)

         # shift and sort
         shifted_scores = torch.abs((-normalized) - ent)
         sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
         sorted_logits = scores.gather(-1, sorted_indices)
         cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

         # Remove tokens with cumulative mass above the threshold
         last_ind = (cumulative_probs < self.mass).sum(dim=1)
         last_ind[last_ind < 0] = 0
         sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
         if self.min_tokens_to_keep > 1:
             # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
             sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
         indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

         scores = scores.masked_fill(indices_to_remove, self.filter_value)
         return scores


class NoisyChannelConfig(PretrainedConfig):
    model_type = "noisy_channel"
    is_composition = True

    def __init__(
        self,
        direct_model_tokenizer_name_or_path=None,
        direct_model_name_or_path=None,
        language_model_tokenizer_name_or_path=None,
        language_model_name_or_path=None,
        channel_model_tokenizer_name_or_path=None,
        channel_model_name_or_path=None,
        lm_scaling_factor=0.1,
        cm_scaling_factor=0.1,
        length_penalty=0.1,
        vocab_size=None,
        is_encoder_decoder=True,
        prefix=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
        dataset_split="train",
        reduce_loss=False,
        label_smoothing=0.0,
        exclude_bos_score=False,
        use_cache=False,
        forced_eos_token_id=None,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            prefix=prefix,
            vocab_size=vocab_size,
            **kwargs,
        )

        self.direct_model_name_or_path = direct_model_name_or_path
        self.direct_model_tokenizer_name_or_path = direct_model_tokenizer_name_or_path
        self.language_model_name_or_path = language_model_name_or_path
        self.language_model_tokenizer_name_or_path = language_model_tokenizer_name_or_path
        self.channel_model_name_or_path = channel_model_name_or_path
        self.channel_model_tokenizer_name_or_path = channel_model_tokenizer_name_or_path
        self.lm_scaling_factor = lm_scaling_factor
        self.cm_scaling_factor = cm_scaling_factor
        self.length_penalty = length_penalty

        self.reduce_loss = reduce_loss
        self.label_smoothing = label_smoothing
        self.exclude_bos_score = exclude_bos_score

        self.use_cache = use_cache


class NoisyChannelModel(torch.nn.Module):
    def __init__(
        self,
        config,
        direct_model,
        channel_model,
        language_model,
    ):
        super().__init__()
        self.config = config
        self.direct_model = direct_model
        self.channel_model = channel_model
        self.language_model = language_model
        self.lm_scaling_factor = self.config.lm_scaling_factor
        self.cm_scaling_factor = self.config.cm_scaling_factor
        self.length_penalty = self.config.length_penalty


class PretrainedNoisyChannelModel(PreTrainedModel):
    config_class = NoisyChannelConfig

    def __init__(
        self,
        config: NoisyChannelConfig,
        direct_model: PreTrainedModel = None,
        channel_model: PreTrainedModel = None,
        language_model: PreTrainedModel = None,
    ):
        super().__init__(config)

        self.config = config
        if direct_model is None:
            direct_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.direct_model_name_or_path)
        self.direct_model = direct_model

        if channel_model is None:
            channel_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.channel_model_name_or_path)
        self.channel_model = channel_model

        if language_model is None:
            language_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.language_model_name_or_path)
        self.language_model = language_model

        self.model = NoisyChannelModel(config, direct_model, channel_model, language_model)
        self.cm_scaling_factor = config.cm_scaling_factor
        self.lm_scaling_factor = config.lm_scaling_factor
        self.length_penalty = config.length_penalty

    def save_pretrained(self, save_directory):
        r"""
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedRagModel.from_pretrained`` class method.
        Arguments:
            save_directory (:obj:`str`):
                Base directory to which to save. Will be created if it doesn't exist. The generator model
                will be saved to save_directory/generator directory. The question encoder model will be saved
                to save_directory/genquestion_encoder directory.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)
        config = copy.deepcopy(self.config)
        config.save_pretrained(save_directory)


class NoisyChannelRerankingModelForConditionalGeneration(PretrainedNoisyChannelModel):

    def __init__(self, config, direct_model, channel_model, language_model):
        super().__init__(config, direct_model, channel_model, language_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.language_model_tokenizer_name_or_path)
        if torch.cuda.device_count() == 2:
            self.is_parallelizable = True
            self.model_parallel = True

            self.direct_model.to(torch.device('cuda:0'))
            self.language_model.to(torch.device('cuda:1'))
            self.channel_model.to(torch.device('cuda:2'))

        else:
            self.direct_model.to(torch.device('cuda:0'))
            self.language_model.to(torch.device('cuda:0'))
            self.channel_model.to(torch.device('cuda:0'))

    def forward(
        self,
        input_ids,
        lm_input_ids=None,
        lm_attention_mask=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        cm_labels=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **unused,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> # Mask filling only works for bart-large
            >>> from transformers import BartTokenizer, BartForConditionalGeneration
            >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
            >>> # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        dm_encoder_outputs, cm_encoder_outputs, lm_encoder_outputs = tuple(encoder_outputs.last_hidden_states)
        dm_logits = self.direct_model(
            input_ids,
            attention_mask=attention_mask.to(self.direct_model.device),
            decoder_input_ids=decoder_input_ids.to(self.direct_model.device),
            encoder_outputs=dm_encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=None,
            use_cache=False, 
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).logits[:, -1, :].to(self.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(dm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=dm_logits,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None
        )


    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, lm_attention_mask=None, **kwargs
    ):
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "lm_input_ids": None, # encoder_outputs is defined. lm_input_ids not needed
            "lm_attention_mask": lm_attention_mask
        }

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs, model_input_name
    ):
        lm_input_ids = model_kwargs.pop("lm_input_ids")
        cm_input_ids = lm_input_ids
        dm_attention_mask = model_kwargs.pop("attention_mask")
        lm_attention_mask = model_kwargs.pop("lm_attention_mask")
        cm_labels = model_kwargs.pop("cm_labels")
        cm_attention_mask = lm_attention_mask
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder_outputs = []
            for model, model_input_ids, attention_mask in zip(
                [self.direct_model, self.channel_model, self.language_model],
                [input_ids, cm_input_ids, lm_input_ids],
                [dm_attention_mask, cm_attention_mask, lm_attention_mask],
            ):
                encoder = model.get_encoder()
                encoder_kwargs = {}
                for argument, value in model_kwargs.items():
                    if not any([argument.startswith(prefix) for prefix in ["decoder_", "cross_attn", "use_cache"]]):
                        if hasattr(value, "to"):
                            encoder_kwargs[argument] = value.to(model.device)
                        else:
                            encoder_kwargs[argument] = value
                encoder_outputs.append(encoder(model_input_ids.to(model.device), attention_mask=attention_mask.to(model.device), return_dict=True, **encoder_kwargs))
            model_kwargs["encoder_outputs"]: CustomModelOutput = CustomModelOutput(last_hidden_states=encoder_outputs)
            model_kwargs["attention_mask"] = dm_attention_mask
            model_kwargs["lm_attention_mask"] = lm_attention_mask
            model_kwargs["lm_input_ids"] = lm_input_ids
            model_kwargs["cm_labels"] = cm_labels
        return model_kwargs

    def __repr__(self):
        return " ".join([model.config._name_or_path for model in [self.direct_model, self.language_model]])

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        lm_attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if lm_attention_mask is not None:
            model_kwargs["lm_attention_mask"] = lm_attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            for i, encoder_output in enumerate(encoder_outputs.last_hidden_states):
                encoder_outputs.last_hidden_states[i]["last_hidden_state"] = encoder_output.last_hidden_state.index_select(
                    0, expanded_return_idx.to(encoder_output.last_hidden_state.device)
                )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs
        

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            beam_scorer (:obj:`BeamScorer`):
                An derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
                An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
                :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.
            max_length (:obj:`int`, `optional`, defaults to 20):
                **DEPRECATED**. Use :obj:`logits_processor` or :obj:`stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utilsBeamSearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.


        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForSeq2SeqLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    BeamSearchScorer,
            ... )
            >>> import torch

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

            >>> encoder_input_str = "translate English to German: How old are you?"
            >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


            >>> # lets run beam search using 3 beams
            >>> num_beams = 3
            >>> # define decoder start token ids
            >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
            >>> input_ids = input_ids * model.config.decoder_start_token_id

            >>> # add encoder_outputs to model keyword arguments
            >>> model_kwargs = {
            ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
            ... }

            >>> # instantiate beam scorer
            >>> beam_scorer = BeamSearchScorer(
            ...     batch_size=1,
            ...     num_beams=num_beams,
            ...     device=model.device,
            ... )

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        beam_scorer.num_beam_hyps_to_keep = beam_scorer.num_beams
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
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
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        
        # prepare inputs for response generation model and channel model
        lm_encoder_outputs = tuple(model_kwargs["encoder_outputs"].last_hidden_states)[0]
        lm_input_ids = torch.repeat_interleave(model_kwargs["lm_input_ids"], beam_scorer.num_beams, dim=0)
        cm_labels = torch.repeat_interleave(model_kwargs["cm_labels"], beam_scorer.num_beams, dim=0)
        cm_labels_final = []
        for cm_label in cm_labels:
            cm_label = cm_label.to(self.channel_model.device)
            cm_label[cm_label == 1] = -100
            cm_labels_final.append(cm_label)
        cm_labels_final = torch.stack(cm_labels_final).to(self.channel_model.device)

        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

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
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=-100,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        # 1. prepare channel model inputs (includes decoded beams)
        # 2. calculate channel model and response generation model scores
        # 3. Rerank
        cm_input_ids = []
        for label, lm_input in zip(sequence_outputs["sequences"], lm_input_ids):
            # u_{T+1}
            new_label = label.clone()
            new_label[new_label == -100] = self.tokenizer.pad_token_id
            # (u_1, ..., u_T, u_{T+1})
            cm_input = torch.cat([
                lm_input[:(lm_input == 1).nonzero(as_tuple=True)[0]].cpu(),
                # last turn is always taken by the system
                torch.tensor(self.tokenizer.convert_tokens_to_ids(["<system>"]) + self.tokenizer(self.tokenizer.batch_decode([new_label.cpu()], skip_special_tokens=True)[0], add_special_tokens=False)["input_ids"] + self.tokenizer.convert_tokens_to_ids(["</s>"]))
            ]).to(self.channel_model.device)
            cm_input_ids.append(cm_input)


        max_len = max(cm_input.shape[-1] for cm_input in cm_input_ids)
        for i, cm_input in enumerate(cm_input_ids):
            difference = max_len - cm_input.shape[-1]
            padding = torch.ones(difference, dtype=torch.int64).to(self.channel_model.device) * -100
            cm_input = torch.cat((cm_input, padding))
            cm_input_ids[i] = cm_input

        cm_input_ids = torch.stack(cm_input_ids).to(self.channel_model.device)
        cm_attention_mask = torch.ones(cm_input_ids.shape, dtype=torch.int64)

        cm_attention_mask[cm_input_ids == -100] = 0
        cm_input_ids[cm_input_ids == -100] = self.tokenizer.pad_token_id


        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        new_input_ids = sequence_outputs["sequences"]

        lm_logits = self.language_model(input_ids=None, encoder_outputs=lm_encoder_outputs, labels=new_input_ids).logits
        lm_scores = -loss_fct(lm_logits.view(-1, self.config.vocab_size), new_input_ids.view(-1)).view(batch_size * beam_scorer.num_beams, new_input_ids.shape[-1])
        del lm_logits
        lm_scores = torch.sum(lm_scores, dim=-1)

        
        cm_logits = self.channel_model(cm_input_ids, labels=cm_labels_final, attention_mask=cm_attention_mask.to(self.channel_model.device)).logits
        cm_scores = -loss_fct(cm_logits.view(-1, self.config.vocab_size), cm_labels_final.view(-1)).view(batch_size * beam_scorer.num_beams, cm_labels_final.shape[-1]) 
        del cm_logits
        cm_scores = torch.sum(cm_scores, dim=-1)

        sequence_outputs["sequences"][sequence_outputs["sequences"] == -100] = self.tokenizer.pad_token_id

        sequence_outputs["sequence_scores"] = sequence_outputs["sequence_scores"] + torch.mul(cm_scores, self.cm_scaling_factor) + torch.mul(lm_scores, self.lm_scaling_factor)
        final_outputs = {"sequences": [], "sequence_scores": []}

        for i in range(batch_size):
            sequences = sequence_outputs["sequences"][i*beam_scorer.num_beam_hyps_to_keep:(i+1)*beam_scorer.num_beam_hyps_to_keep]
            sequence_scores = sequence_outputs["sequence_scores"][i*beam_scorer.num_beam_hyps_to_keep:(i+1)*beam_scorer.num_beam_hyps_to_keep]
            zipped = list(zip(sequences, sequence_scores))
            sorted_sequences = sorted(zipped, key=lambda x: x[1], reverse=True)
            final_outputs["sequences"].append(sorted_sequences[0][0])
            final_outputs["sequence_scores"].append(sorted_sequences[0][1])

        for key in ["sequences", "sequence_scores"]:
            final_outputs[key] = torch.stack(final_outputs[key])

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=final_outputs["sequences"],
                    sequences_scores=final_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=final_outputs["sequences"],
                    sequences_scores=final_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return final_outputs["sequences"]
        

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        direct_model = AutoModelForSeq2SeqLM.from_pretrained(config.direct_model_name_or_path)
        channel_model = AutoModelForSeq2SeqLM.from_pretrained(config.channel_model_name_or_path)
        language_model = AutoModelForSeq2SeqLM.from_pretrained(config.language_model_name_or_path)
        return cls(config, direct_model, channel_model, language_model,)# config.cm_scaling_factor, config.lm_scaling_factor)

    def resize_token_embeddings(self, new_num_tokens):
        for model in [self.direct_model, self.language_model]:
            model.resize_token_embeddings(new_num_tokens)
