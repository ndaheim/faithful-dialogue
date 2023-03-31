import numpy as np
from datasets import load_metric
from transformers import PretrainedConfig, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, DefaultDataCollator

from dialog.methods.base import Method
from dialog.methods.preprocessing.generation import DocumentGroundedPreprocessor, Seq2SeqPreprocessor, DocumentGroundedPreprocessorWithKnowledgeLabels, DocumentGroundedPreprocessorWithCTRLTokens, DocumentGroundedPreprocessorForDensityRatio, \
 DocumentGroundedPreprocessorForNoisyChannel
from dialog.methods.trainer_fim import CustomSeq2SeqTrainerForFisherCalculation
from dialog.methods.trainer_seq2seq import CustomSeq2SeqTrainer, DataCollatorWithLMInputs, DataCollatorForNoisyChannel

from dialog.models.dexperts import DensityRatioMethodModelForConditionalGeneration
from dialog.models.noisy_channel import NoisyChannelRerankingModelForConditionalGeneration


class Seq2SeqMethod(Method):
    name = "seq2seq"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = [
            load_metric(metric, cache_dir=self.model_args.cache_dir) for metric in ["sacrebleu"]
        ]

    def compute_metrics(self, p):
        p.label_ids[p.label_ids == -100] = self.tokenizer.pad_token_id
        p.predictions[p.predictions == -100] = self.tokenizer.pad_token_id
        predictions_strings = self.tokenizer.batch_decode(
            p.predictions, skip_special_tokens=True
        )
        reference_strings = [[ref] for ref in self.tokenizer.batch_decode(
            p.label_ids, skip_special_tokens=True)]

        results = {}

        for metric in self.metrics:
            results.update(
                metric.compute(
                    predictions=predictions_strings,
                    references=reference_strings
                )
            )
        return results


    def get_trainer_class(self):
        return CustomSeq2SeqTrainer

    def get_data_collator(self):
        return DataCollatorForSeq2Seq(self.tokenizer)

    def get_model_class(self, config: PretrainedConfig):
        return AutoModelForSeq2SeqLM

    def preprocess_features(self, features):
        processor = Seq2SeqPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        if self.data_args.is_training:
            return_dict["labels"] = labels

        return return_dict

    def postprocess_predictions(self, p, dataset):
        model_class = self.get_model_class(self.config)

        p.predictions[p.predictions == -100] = self.tokenizer.pad_token_id
        out = self.tokenizer.batch_decode(
            p.predictions, skip_special_tokens=True
        )

        return out

class DocumentGroundedGenerationMethod(Seq2SeqMethod):
    name = "document_grounded_generation"

    def preprocess_features(self, features):
        processor = DocumentGroundedPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        if self.data_args.is_training:
            return_dict["labels"] = labels

        return return_dict

    def get_special_tokens(self):
        return [
            self.model_args.user_token,
            self.model_args.system_token,
            self.model_args.knowledge_tag_token,
            self.model_args.knowledge_sep_token,
        ]

class DocumentGroundedGenerationWithCTRLMethod(DocumentGroundedGenerationMethod):

    name = "document_grounded_generation_ctrl"

    def preprocess_features(self, features):
        processor = DocumentGroundedPreprocessorWithCTRLTokens(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        if self.data_args.is_training:
            return_dict["labels"] = labels

        return return_dict

    def get_special_tokens(self):
        return [
            self.model_args.user_token,
            self.model_args.system_token,
            self.model_args.knowledge_tag_token,
            self.model_args.knowledge_sep_token,
            "<entailed>", "<non-entailed>", "<low-prec>", "<med-prec>", "<high-prec>"
        ]

class FisherApproximationForDocumentGroundedGenerationMethod(DocumentGroundedGenerationMethod):
    name = "fisher_approx_document_grounded_generation"

    def get_trainer_class(self):
        return CustomSeq2SeqTrainerForFisherCalculation
    
class FisherApproximationForDocumentGroundedGenerationWithCTRLMethod(DocumentGroundedGenerationWithCTRLMethod):
    name = "fisher_approx_document_grounded_generation_ctrl"

    def get_trainer_class(self):
        return CustomSeq2SeqTrainerForFisherCalculation

class DocumentGroundedGenerationWithDensityRatioMethod(DocumentGroundedGenerationWithCTRLMethod):
    # DExperts
    name = "document_grounded_generation_density_ratio"

    def preprocess_features(self, features):
        processor = DocumentGroundedPreprocessorForDensityRatio(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, lm_input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
            "lm_input_ids": lm_input_ids
        }

        if self.data_args.is_training:
            return_dict["labels"] = labels

        return return_dict

    def get_data_collator(self):
        return DataCollatorWithLMInputs(self.tokenizer)

    def get_model_class(self, config):
        return DensityRatioMethodModelForConditionalGeneration

class ChannelModelMethod(DocumentGroundedGenerationMethod):

    name = "channel_model"

    def preprocess_features(self, features):
        for i, (response, context, knowledge) in enumerate(zip(features["response"], features["context"], features["knowledge"])):
            features["context"][i].append({"user": "S", "text": response, "dialog_act": ""})
            features["response"][i] = knowledge[0]["text"]
            features["knowledge"][i] = []

        processor = DocumentGroundedPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            'input_ids': input_ids,
            'labels': labels
        }

        return return_dict

class ResponseGenerationMethod(Seq2SeqMethod):
    name = "response_generation"

    def preprocess_features(self, features):
        for i in range(len(features["knowledge"])):
            features["knowledge"][i] = []

        processor = DocumentGroundedPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        if self.data_args.is_training:
            return_dict["labels"] = labels

        return return_dict

class NoisyChannelModelMethod(DocumentGroundedGenerationMethod):

    name = "noisy_channel_reranking"

    def get_model_class(self, config: PretrainedConfig):
        return NoisyChannelRerankingModelForConditionalGeneration

    def preprocess_features(self, features):
        processor = DocumentGroundedPreprocessorForNoisyChannel(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, lm_input_ids, labels = processor.preprocess(features)

        for i, (response, context, knowledge) in enumerate(zip(features["response"], features["context"], features["knowledge"])):
            # needed to evaluate p(K|u_{T+1}, u_T) in the channel model
            features["response"][i] = knowledge[0]["text"]
            features["knowledge"][i] = []

        processor = DocumentGroundedPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        _, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
            "lm_input_ids": lm_input_ids,
            "cm_labels": labels
        }

        return return_dict

    def get_data_collator(self):
        return DataCollatorForNoisyChannel(self.tokenizer)