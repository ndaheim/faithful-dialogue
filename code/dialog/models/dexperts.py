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
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import LogitsProcessorList
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


class EnsembleModelOutput(ModelOutput):
    last_hidden_states = None


class DensityRatioMethodConfig(PretrainedConfig):
    model_type = "density_ratio_method"
    is_composition = True

    def __init__(
        self,
        direct_model_tokenizer_name_or_path=None,
        direct_model_name_or_path=None,
        language_model_tokenizer_name_or_path=None,
        language_model_name_or_path=None,
        internal_language_model_tokenizer_name_or_path=None,
        internal_language_model_name_or_path=None,
        lm_scaling_factor=0.1,
        ilm_scaling_factor=0.1,
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
        self.internal_language_model_name_or_path = internal_language_model_name_or_path
        self.internal_language_model_tokenizer_name_or_path = internal_language_model_tokenizer_name_or_path
        self.lm_scaling_factor = lm_scaling_factor
        self.ilm_scaling_factor = ilm_scaling_factor

        self.reduce_loss = reduce_loss
        self.label_smoothing = label_smoothing
        self.exclude_bos_score = exclude_bos_score

        self.use_cache = use_cache


class DensityRatioMethodModel(torch.nn.Module):
    def __init__(
        self,
        config,
        direct_model,
        internal_language_model,
        language_model,
    ):
        super().__init__()
        self.config = config
        self.direct_model = direct_model
        self.internal_language_model = internal_language_model
        self.language_model = language_model
        self.lm_scaling_factor = self.config.lm_scaling_factor
        self.ilm_scaling_factor = self.config.ilm_scaling_factor


class PretrainedDensityRatioMethodModel(PreTrainedModel):
    config_class = DensityRatioMethodConfig

    def __init__(
        self,
        config: DensityRatioMethodConfig,
        direct_model: PreTrainedModel = None,
        internal_language_model: PreTrainedModel = None,
        language_model: PreTrainedModel = None,
    ):
        super().__init__(config)

        self.config = config
        if direct_model is None:
            direct_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.direct_model_name_or_path)
        self.direct_model = direct_model

        if internal_language_model is None:
            internal_language_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.internal_language_model_name_or_path)
        self.internal_language_model = internal_language_model

        if language_model is None:
            language_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.language_model_name_or_path)
        self.language_model = language_model

        self.model = DensityRatioMethodModel(config, direct_model, internal_language_model, language_model)
        self.ilm_scaling_factor = config.ilm_scaling_factor
        self.lm_scaling_factor = config.lm_scaling_factor

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
        direct_model_output_dir = os.path.join(save_directory, "direct_model")
        self.model.direct_model.save_pretrained(direct_model_output_dir)
        internal_language_model_output_dir = os.path.join(save_directory, "internal_language_model")
        self.model.internal_language_model.save_pretrained(internal_language_model_output_dir)
        language_model_output_dir = os.path.join(save_directory, "language_model")
        self.model.language_model.save_pretrained(language_model_output_dir)
        config = copy.deepcopy(self.config)
        config.pretrained_direct_model_name_or_path = direct_model_output_dir
        config.pretrained_internal_language_model_name_or_path = internal_language_model_output_dir
        config.pretrained_language_model_name_or_path = language_model_output_dir
        config.pretrained_direct_model_tokenizer = config.direct_model_tokenizer_name_or_path
        config.vocab_size = self.model.language_model.config.vocab_size
        config.save_pretrained(save_directory)


class DensityRatioMethodModelForConditionalGeneration(PretrainedDensityRatioMethodModel):

    def __init__(self, config, direct_model, internal_language_model, language_model, scaling_factor=.3):
        super().__init__(config, direct_model, internal_language_model, language_model)

        if torch.cuda.device_count() == 2:
            self.is_parallelizable = True
            self.model_parallel = True

            self.direct_model.to(torch.device('cuda:0'))
            self.language_model.to(torch.device('cuda:1'))
            self.internal_language_model.to(torch.device('cuda:2'))

    def forward(
        self,
        input_ids,
        lm_input_ids=None,
        lm_attention_mask=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_chache=None,
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

        bound = min([model.config.vocab_size for model in [self.direct_model, self.internal_language_model, self.language_model]])

        dm_encoder_outputs, ilm_encoder_outputs, lm_encoder_outputs = tuple(encoder_outputs.last_hidden_states)
        dm_logits = self.direct_model(
            None,
            attention_mask=attention_mask.to(self.direct_model.device),
            decoder_input_ids=decoder_input_ids.to(self.direct_model.device),
            encoder_outputs=dm_encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=None,
            use_cache=False, # TODO: check
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).logits[:, -1, :bound].to(self.device)

        ilm_logits = self.internal_language_model(
            None,
            attention_mask=attention_mask.to(self.internal_language_model.device),
            decoder_input_ids=decoder_input_ids.to(self.internal_language_model.device),
            encoder_outputs=ilm_encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).logits[:, -1, :bound].to(self.device)

        lm_logits = self.language_model(
            None,
            attention_mask=lm_attention_mask.to(self.language_model.device),
            decoder_input_ids=decoder_input_ids.to(self.language_model.device),
            encoder_outputs=lm_encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).logits[:, -1, :bound].to(self.device)
        dm_scores, ilm_scores, lm_scores = tuple(F.log_softmax(logits, dim=-1) for logits in [dm_logits, ilm_logits, lm_logits])

        interpolated_scores = dm_scores - torch.mul(ilm_scores, self.ilm_scaling_factor) + torch.mul(lm_scores, self.lm_scaling_factor)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=interpolated_scores,
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
            "lm_input_ids": kwargs["lm_input_ids"], # encoder_outputs is defined. lm_input_ids not needed
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
        ilm_input_ids = input_ids
        dm_attention_mask = model_kwargs.pop("attention_mask")
        lm_attention_mask = model_kwargs.pop("lm_attention_mask")
        ilm_attention_mask = dm_attention_mask

        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder_outputs = []
            for i, (model, model_input_ids, attention_mask) in enumerate(zip(
                [self.direct_model, self.internal_language_model, self.language_model],
                [input_ids, ilm_input_ids, lm_input_ids],
                [dm_attention_mask, ilm_attention_mask, lm_attention_mask],
            )):
                encoder = model.get_encoder()
                encoder_kwargs = {}
                for argument, value in model_kwargs.items():
                    if not argument.startswith("decoder_"):
                        if hasattr(value, "to"):
                            encoder_kwargs[argument] = value.to(model.device)
                        else:
                            encoder_kwargs[argument] = value
                encoder_kwargs.pop("use_cache")
                encoder_outputs.append(encoder(model_input_ids.to(model.device), attention_mask=attention_mask.to(model.device), return_dict=True, **encoder_kwargs))
            model_kwargs["encoder_outputs"]: EnsembleModelOutput = EnsembleModelOutput(last_hidden_states=encoder_outputs)
            model_kwargs["attention_mask"] = dm_attention_mask
            model_kwargs["lm_attention_mask"] = lm_attention_mask
            model_kwargs["lm_input_ids"] = lm_input_ids
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
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
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
                    cross_attentions=cross_attentions,
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        direct_model = AutoModelForSeq2SeqLM.from_pretrained(config.direct_model_name_or_path)
        internal_language_model = AutoModelForSeq2SeqLM.from_pretrained(config.internal_language_model_name_or_path)
        language_model = AutoModelForSeq2SeqLM.from_pretrained(config.language_model_name_or_path)
        return cls(config, direct_model, internal_language_model, language_model)

    def resize_token_embeddings(self, new_num_tokens):
        for model in [self.direct_model, self.language_model]:
            model.resize_token_embeddings(new_num_tokens)

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
        r"""
        Generates sequences for models with a language modeling head using beam search with multinomial sampling.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (:obj:`BeamScorer`):
                A derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
                An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
                :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.
            logits_warper (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsWarper` used to warp the prediction score distribution of the language
                modeling head applied before multinomial sampling at each generation step.
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
            :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ...     AutoTokenizer,
            ...     AutoModelForSeq2SeqLM,
            ...     LogitsProcessorList,
            ...     MinLengthLogitsProcessor,
            ...     TopKLogitsWarper,
            ...     TemperatureLogitsWarper,
            ...     BeamSearchScorer,
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
            ...     max_length=model.config.max_length,
            ...     num_beams=num_beams,
            ...     device=model.device,
            ... )

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)
            ... ])
            >>> # instantiate logits processors
            >>> logits_warper = LogitsProcessorList([
            ...     TopKLogitsWarper(50),
            ...     TemperatureLogitsWarper(0.7),
            ... ])

            >>> outputs = model.beam_sample(
            ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
            ... )

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
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

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
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
            next_token_scores = logits_warper(input_ids, next_token_scores)

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

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

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
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSampleEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSampleDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]