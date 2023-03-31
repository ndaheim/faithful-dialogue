# http://parl.ai/downloads/fits/fits_data_v0.1.tar.gz 

import abc

import datasets

_CITATION = ""
_DESCRIPTION = ""
_HOMEPAGE = ""

import json
import logging

import datasets
import numpy as np
import spacy
import transformers
from accelerate import Accelerator
from datasets import Dataset
from nltk.tokenize import RegexpTokenizer
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
tokenizer = RegexpTokenizer(r'\w+')

class CTRLMixin(object):
    
    _ENTAILMENT_TOKEN_MAP = {
        0: "<non-entailed>",
        1: "<non-entailed>",
        2: "<entailed>"
    }
    
    _LEXICAL_TOKEN_MAP = {
        0: "<low-prec>",
        1: "<med-prec>",
        2: "<high-prec>"
    }

    _COVERAGE_TOKEN_MAP = {
        0: "<low-coverage>",
        1: "<med-coverage>",
        2: "<high-coverage>"
    }

    _DENSITY_TOKEN_MAP = {
        0: "<low-density>",
        1: "<med-density>",
        2: "<high-density>"
    }

    def coverage(self, pred, F):
        if len(pred) > 0:
            return sum([len(f) for f in F]) / len(pred)
        else:
            return 0

    def density(self, pred, F):
        if len(pred) > 0:
            return sum([len(f)**2 for f in F]) / len(pred)
        else:
            return 0
    def lcs(self, pred, F):
        if len(pred) > 0:
            return max([len(f) for f in F]) / len(pred)
        else:
            return 0

    def calculate_F(self, pred, ref, tokenizer):
        pred = tokenizer.tokenize(pred)
        ref = tokenizer.tokenize(ref)
        F = []
        i, j = 0, 0

        while i < len(pred):
            f = []
            while j < len(ref):
                if pred[i] == ref[j]:
                    i_prime, j_prime = i, j
                    while i_prime < len(pred) and j_prime < len(ref) and pred[i_prime] == ref[j_prime]:
                        i_prime, j_prime = i_prime+1, j_prime+1
                    if len(f) < i_prime - i:
                        f = pred[i: i_prime]
                    j = j_prime
                else:
                    j = j + 1
            i, j = i + max(len(f), 1), 1
            F.append(f)
        return F, pred

    def _tokenize(self, text, as_string=False):
        tokens = tokenizer.tokenize(text)
        if as_string:
            return " ".join(tokens)
        else:
            return tokens
        
    def _compute_lexical_overlap_group(self, lexical_overlaps):
        lexical_overlaps = np.array(lexical_overlaps)
        sorted_lex_indices = np.argsort(lexical_overlaps)
        lo_indices, med_indices, _ = np.array_split(sorted_lex_indices, 3)
        max_lo_overlap = lexical_overlaps[lo_indices[-1]] if lo_indices.size > 0 else 0
        max_med_overlap = lexical_overlaps[med_indices[-1]] if med_indices.size > 0 else 0

        groups = [-1] * len(lexical_overlaps)

        for idx in range(len(lexical_overlaps)):
            if lexical_overlaps[idx] <= max_lo_overlap:
                groups[idx] = 0
            elif max_lo_overlap < lexical_overlaps[idx] <= max_med_overlap:
                groups[idx] = 1
            else:
                groups[idx] = 2

        return groups

    def _measure_lexical_overlap(self, tokens, ref_tokens):
        """
        Noted in https://aclanthology.org/2021.acl-long.58/:
        "this may not reflect semantic differences in the information being shared
        (e.g. dropping the word ‘not’ may yield high lexical precision but a very different semantic meaning
        from the original evidence)."
        :param tokens: utterance tokens
        :param ref_tokens: reference tokens
        :return: lexical overlap, ratio of common terms over length of tokens
        """
        if not tokens:
            return 0.0

        return sum(1 for t in tokens if t in ref_tokens) / len(tokens)
    
    def _predict_nli_labels(self, model_name_or_path, nli_data, max_length=384, per_device_batch_size=2):
        accelerator = Accelerator()
        logger.info(accelerator.state)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.logging.set_verbosity_error()

        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=3, finetuning_task="mnli")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
        ).to("cuda:0")

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (examples["premise"], examples["hypothesis"])
            result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)

            return result

        raw_dataset = Dataset.from_dict(nli_data)
        processed_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc="Running tokenizer on dataset",
        )

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
        dataloader = DataLoader(processed_dataset, collate_fn=data_collator, batch_size=per_device_batch_size)
        model, dataloader = accelerator.prepare(model, dataloader)
        model.eval()
        for step, batch in enumerate(tqdm(dataloader, total=len(processed_dataset))):
            outputs = model(**batch.to("cuda:0"))
            predictions = accelerator.gather(outputs.logits.argmax(dim=-1)).detach().cpu().tolist()
            yield from predictions

    def _add_abstractiveness_tokens(self, data):
        coverages, densities, lcss = [], [], []
        tokenizer = RegexpTokenizer(r'\w+')
        for samples in data:
            for sample in samples:
                sample["ctrl_tokens"] = ""
                F, pred = self.calculate_F(sample["response"], sample["knowledge"][0]["text"], tokenizer)
                coverages.append(self.coverage(pred, F))
                densities.append(self.density(pred, F))
                lcss.append(self.lcs(pred, F))

        coverage_groups = self._compute_lexical_overlap_group(coverages)
        density_groups = self._compute_lexical_overlap_group(densities)

        group_idx = 0

        for idx, sample in enumerate(data):
            for jdx, new_sample in enumerate(sample):
                sample[jdx]["ctrl_tokens"] += self._COVERAGE_TOKEN_MAP[coverage_groups[group_idx]]
                sample[jdx]["ctrl_tokens"] += self._DENSITY_TOKEN_MAP[density_groups[group_idx]]
                group_idx += 1

        return data
        
    def _add_control_tokens(self, data):
        nli_data = {
            "premise": [],
            "hypothesis": [],
            "did": [],
        }
        
        lexical_overlaps = []
        for idx, sample in tqdm(enumerate(data)):
            for jdx, new_sample in enumerate(sample):
                new_sample["ctrl_tokens"] = ""
                premise = new_sample["knowledge"][0]["text"]
                knowledge_tokens = self._tokenize(premise)
                hypothesis = new_sample["response"]
                response_tokens = self._tokenize(hypothesis)
                
                nli_data["premise"].append(premise)
                nli_data["hypothesis"].append(hypothesis)
                nli_data["did"].append(idx)
                
                lexical_overlap = self._measure_lexical_overlap(response_tokens, knowledge_tokens)
                lexical_overlaps.append(lexical_overlap)
        lexical_overlaps = np.array(lexical_overlaps)
        lexical_groups = self._compute_lexical_overlap_group(lexical_overlaps)
        
        group_idx = 0

        for idx, sample in enumerate(data):
            for jdx, new_sample in enumerate(sample):
                sample[jdx]["ctrl_tokens"] += self._LEXICAL_TOKEN_MAP[lexical_groups[group_idx]]
                group_idx += 1
        
        nli_model = "roberta-large-mnli"
        nli_labels = list(self._predict_nli_labels(nli_model, nli_data))
        
        group_idx = 0
        for idx, sample in enumerate(data):
            for jdx, new_sample in enumerate(sample):
                sample[jdx]["ctrl_tokens"] += f" {self._ENTAILMENT_TOKEN_MAP[nli_labels[group_idx]]}"
                group_idx += 1

        return data

class Seq2SeqDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for Art."""

    def __init__(self, **kwargs):
        """BuilderConfig for Art.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Seq2SeqDatasetConfig, self).__init__(**kwargs)

class Seq2SeqDataset(object):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_CONFIG_NAME = "default"

    BUILDER_CONFIGS = [
        Seq2SeqDatasetConfig(
            name=name,
            version=datasets.Version("1.0.0"),
            description=""
        ) for name in ["response_generation"]
    ]


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "dataset_id": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string")
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        pass

    def _generate_examples(self, filepath):
        pass

    @abc.abstractmethod
    def _map_to_common_format(self, sample):
        pass

    def _download_files(self, urls, data_files, dl_manager):
        if data_files is not None:
            raise NotImplementedError()
        return dl_manager.download_and_extract(urls)  



class DocumentGroundedDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for Art."""

    def __init__(self, **kwargs):
        """BuilderConfig for Art.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DocumentGroundedDatasetConfig, self).__init__(**kwargs)


class DocumentGroundedDataset(CTRLMixin):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_CONFIG_NAME = "default"

    BUILDER_CONFIGS = [
        DocumentGroundedDatasetConfig(
            name=name,
            version=datasets.Version("1.0.0"),
            description=""
        ) for name in ["response_generation", "ctrl", "hallucinated_response", "swap_knowledge", "swap_knowledge_ctrl", "hallucinated_response_ctrl", "cape_expert", "cape_anti_expert", "abstractive_response_generation", "abstractive_response_generation_ctrl"]
    ]


    def _info(self):
        feature_dict = {
            "id": datasets.Value("string"),
            "context": [
                {
                    "text": datasets.Value("string"),
                    "user": datasets.Value("string"),
                    "dialog_act": datasets.Value("string"),
                }
            ],
            "dataset_id": datasets.Value("string"),
            "dialog_act": datasets.Value("string"),
            "knowledge": [{
                "description": datasets.Value("string"),
                "text": datasets.Value("string")
            }],
            "response": datasets.Value("string")
        }

        if "ctrl" in self.config.name or "cape" in self.config.name:
            feature_dict["ctrl_tokens"] = datasets.Value("string")
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(feature_dict),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        pass

    def _generate_examples(self, data):
        if "ctrl" in self.config.name or "cape" in self.config.name:
            data = self._add_control_tokens(data)
        if self.config.name == "abstractive_response_generation":
            data = self._add_abstractiveness_tokens(data)

        idx = 0

        for samples in data:
            lower_bound = idx
            for new_sample in samples:
                if "swap_knowledge" in self.config.name:
                    support = set(range(len(data))) - set(range(lower_bound, lower_bound+len(samples)))
                    new_idx = np.random.choice(list(support))
                    possible_samples = data[new_idx]
                    sample_idx = np.random.choice(range(len(possible_samples)))
                    new_sample["knowledge"] = data[new_idx][sample_idx]["knowledge"]
                new_sample["id"] = str(idx)
                idx += 1

                if self.config.name == "cape_expert" and "<entailed>" not in new_sample["ctrl_tokens"]:
                    continue
                elif self.config.name == "cape_anti_expert" and "<non-entailed>" not in new_sample["ctrl_tokens"]:
                    continue
                elif self.config.name == "abstractive_response_generation":
                    if any([token in new_sample["ctrl_tokens"] for token in ["high_density", "med-density", "low-coverage"]]):
                        continue
                    else:
                        del new_sample["ctrl_tokens"]

                yield idx, new_sample


    def _download_files(self, urls, data_files, dl_manager):
        if data_files is not None:
            self.data_files = {}
            for key, value in data_files.items():
                with open(value[0], "r") as f:
                    self.data_files[key] = json.load(f)
        return dl_manager.download_and_extract(urls)


class UnGroundedDialogDataset(object):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_CONFIG_NAME = "default"

    BUILDER_CONFIGS = [
        DocumentGroundedDatasetConfig(
            name=name,
            version=datasets.Version("1.0.0"),
            description=""
        ) for name in ["response_generation"]
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": [
                        {
                            "text": datasets.Value("string"),
                            "user": datasets.Value("string"),
                            "dialog_act": datasets.Value("string"),
                        }
                    ],
                    "dataset_id": datasets.Value("string"),
                    "dialog_act": datasets.Value("string"),
                    "response": datasets.Value("string")
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        pass

    def _generate_examples(self, filepath):
        pass

    @abc.abstractmethod
    def _map_to_common_format(self, sample):
        pass

    def _download_files(self, urls, data_files, dl_manager):
        if data_files is not None:
            raise NotImplementedError()
        return dl_manager.download_and_extract(urls)
