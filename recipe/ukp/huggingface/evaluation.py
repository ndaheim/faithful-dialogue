import copy
import json
import os
import subprocess as sp

from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed
from i6_core.tools.download import DownloadJob

Path = setup_path(__package__)

class CalculateMetricsJob(Job):
    """
    """

    def __init__(
        self,
        code_root,
        dataset_name,
        split,
        model_output_file,
        *,  # args below are keyword only
        bert_score_model_name_or_path="microsoft/deberta-large-mnli",
        dataset_filter_dict=None,
        time_rqmt=3,
        mem_rqmt=12,
        cpu_rqmt=1,
        gpu_rqmt=1,
        python_exe=None,
        **kwargs,
    ):
        """
        :param code_root: Root directory for the training scripts. Expected to contain a training script.
        :param config:
        :param num_epochs:
        :param time_rqmt:
        :param mem_rqmt:
        :param cpu_rqmt:
        :param gpu_rqmt:
        """

        self.code_root = code_root
        self.dataset = dataset_name
        self.dataset_filter_dict = dataset_filter_dict
        self.model_output_file = model_output_file
        self.split = split
        self.bert_score_model_name_or_path = bert_score_model_name_or_path

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }
        self.out_results_file = self.output_path("metrics.json")

    def get_tokens(self, text, nlp):
        doc = nlp(text)
        tokens = [tok.text.lower()
                for tok in doc if not tok.is_stop and not tok.is_punct]
        return tokens

    def f1_score(self, gold, pred, nlp):
        from collections import Counter
        gold_toks = self.get_tokens(gold, nlp)
        pred_toks = self.get_tokens(pred, nlp)

        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def run(self):
        import json
        import os
        os.environ["TRANSFORMERS_CACHE"] = gs.TRANSFORMERS_CACHE
        os.environ["HF_HOME"] = gs.HF_HOME
        
        import bert_score
        import numpy as np
        import spacy
        import torch
        from datasets import load_dataset
        from nltk.tokenize import RegexpTokenizer
        from sacrebleu.metrics import BLEU
        from torch.nn.functional import log_softmax
        from tqdm import tqdm
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
        
        nlp = spacy.load("en_core_web_sm")
        regexp_tokenizer = RegexpTokenizer(r'\w+')

        def coverage(F, pred):
            if len(pred) > 0:
                return sum([len(f) for f in F]) / len(pred)
            else:
                return 0

        def density(F, pred):
            if len(pred) > 0:
                return sum([len(f)**2 for f in F]) / len(pred)
            else:
                return 0
        def lcs(F, pred):
            if len(pred) > 0:
                return max([len(f) for f in F]) / len(pred)
            else:
                return 0

        def calculate_F(pred, ref, tokenizer):
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

        tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/roberta-large-faithcritic", return_tensors="pt")
        model = AutoModelForSequenceClassification.from_pretrained("McGill-NLP/roberta-large-faithcritic").to("cuda:0")

        with open(self.model_output_file, "r") as f:
            predictions = json.load(f)

        dataset = load_dataset(self.dataset, "response_generation", split=self.split)

        critic_scores, densities, coverages, lcss = [], [], [], []

        for prediction, ground_truth in tqdm(zip(predictions, dataset)):
            F, pred = calculate_F(prediction, ground_truth["knowledge"][0]["text"], regexp_tokenizer)
            coverages.append(coverage(F, pred))
            densities.append(density(F, pred))
            lcss.append(lcs(F, pred))

            with torch.no_grad():
                input_ = tokenizer(ground_truth["knowledge"][0]["text"], prediction, return_tensors="pt", truncation=True, max_length=256).to(model.device)
                critic_scores.append(torch.argmax(model(**input_).logits).cpu().detach().numpy())

        all_knowledge = [sample["knowledge"][0]["text"] for sample in dataset]
        all_refs = [sample["response"] for sample in dataset]
        bert_scores_knowledge = bert_score.score(predictions, all_knowledge, model_type=self.bert_score_model_name_or_path, batch_size=4)
        bert_scores_gen = bert_score.score(predictions, all_refs, model_type=self.bert_score_model_name_or_path)
        refs_sbleu = [[sample["response"] for sample in dataset]]

        bleu = BLEU()
        bleu_score = bleu.corpus_score(predictions, refs_sbleu)
        critic_score = round(np.mean(critic_scores), 4)
        P_k, R_k, f1_k = (tensor.mean().item() for tensor in bert_scores_knowledge)
        P_g, R_g, f1_g = (tensor.mean().item() for tensor in bert_scores_gen)

        scores = {
            "sacrebleu": bleu_score.score,
            "critic_score": critic_score,
            "BertScore": {
                "generation": {
                    "P": P_g,
                    "R": R_g,
                    "F-1": f1_g  
                },
                "knowledge": {
                    "P": P_k,
                    "R": R_k,
                    "F-1": f1_k
                },
                "model": self.bert_score_model_name_or_path,
            },
            "knowledge_f1": round(np.mean([self.f1_score(pred, ref, nlp) for pred, ref in tqdm(zip(predictions, all_knowledge))]), 4),
            "coverage": round(np.mean(coverages), 4),
            "density": round(np.mean(densities), 4),
            "lcs": round(np.mean(lcss), 4),
        }


        with open(self.out_results_file, "w") as f:
            json.dump(
                scores,
                f
            )

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=False)

    def update_rqmt_pure(self, **kwargs):
        rqmt = self.rqmt.copy()
        
        for key in rqmt.keys():
            value = kwargs.get(key, None)
            if value is not None:
                rqmt[key] = value

        return rqmt

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)

class CalculateMetricsForMultiDocJob(CalculateMetricsJob):

    def run(self):
        import itertools
        import json
        import os
        os.environ["TRANSFORMERS_CACHE"] = gs.TRANSFORMERS_CACHE
        os.environ["HF_HOME"] = gs.HF_HOME
        
        import bert_score
        import numpy as np
        import spacy
        import torch
        from datasets import load_dataset
        from nltk.tokenize import RegexpTokenizer
        from sacrebleu.metrics import BLEU
        from tqdm import tqdm
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        nlp = spacy.load("en_core_web_sm")
        regexp_tokenizer = RegexpTokenizer(r'\w+')

        def coverage(F, pred):
            if len(pred) > 0:
                return sum([len(f) for f in F]) / len(pred)
            else:
                return 0

        def density(F, pred):
            if len(pred) > 0:
                return sum([len(f)**2 for f in F]) / len(pred)
            else:
                return 0
        def lcs(F, pred):
            if len(pred) > 0:
                return max([len(f) for f in F]) / len(pred)
            else:
                return 0

        def calculate_F(pred, ref, tokenizer):
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

        tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/roberta-large-faithcritic", return_tensors="pt")
        model = AutoModelForSequenceClassification.from_pretrained("McGill-NLP/roberta-large-faithcritic").to("cuda:0")

        with open(self.model_output_file, "r") as f:
            predictions = json.load(f)

        dataset = load_dataset(self.dataset, "response_generation", split=self.split)

        critic_scores, densities, coverages, lcss = [], [], [], []

        for prediction, ground_truth in tqdm(zip(predictions, dataset)):
            local_lcss, local_coverages, local_critic_scores, local_densities = [], [], [], []
            for i in range(len(ground_truth["knowledge"])):
                F, pred = calculate_F(prediction, ground_truth["knowledge"][i]["text"], regexp_tokenizer)
                local_coverages.append(coverage(F, pred))
                local_densities.append(density(F, pred))
                local_lcss.append(lcs(F, pred))
                coverages.append(np.max(local_coverages))
                densities.append(np.min(local_densities))
                lcss.append(np.min(local_lcss))

                with torch.no_grad():
                    input_ = tokenizer(ground_truth["knowledge"][i]["text"], prediction, return_tensors="pt", truncation=True, max_length=256).to(model.device)
                    local_critic_scores.append(torch.argmax(model(**input_).logits).cpu().detach().numpy())
                    critic_scores.append(np.min(local_critic_scores))

        all_knowledge = []
        all_preds = []
        for sample, pred in zip(dataset, predictions):
            for snippet in sample["knowledge"]:
                all_preds.append(pred)
                all_knowledge.append(snippet["text"])

        all_refs = [sample["response"] for sample in dataset]
        bert_scores_knowledge = bert_score.score(all_preds, all_knowledge, model_type=self.bert_score_model_name_or_path)
        bert_scores_gen = bert_score.score(predictions, all_refs, model_type=self.bert_score_model_name_or_path)
        refs_sbleu = [[sample["response"] for sample in dataset]]

        bleu = BLEU()
        score = bleu.corpus_score(predictions, refs_sbleu)
        critic_score = round(np.mean(critic_scores), 4)
        P_k, R_k, f1_k = (tensor.mean().item() for tensor in bert_scores_knowledge)
        P_g, R_g, f1_g = (tensor.mean().item() for tensor in bert_scores_gen)

        all_knowledge = []
        all_preds = []
        for sample, pred in zip(dataset, predictions):
            new_knowledge = []
            new_preds = []
            for snippet in sample["knowledge"]:
                new_preds.append(pred)
                new_knowledge.append(snippet["text"])
            all_knowledge.append(new_knowledge)
            all_preds.append(new_preds)
                
        scores = {
            "sacrebleu": score.score,
            "critic_score": critic_score,
            "BertScore": {
                "generation": {
                    "P": P_g,
                    "R": R_g,
                    "F-1": f1_g  
                },
                "knowledge": {
                    "P": P_k,
                    "R": R_k,
                    "F-1": f1_k
                },
                "model": self.bert_score_model_name_or_path,
            },
            "knowledge_f1": round(np.mean([np.max([self.f1_score(pred, ref, nlp) for pred, ref in zip(preds, refs)])
                                          for preds, refs in tqdm(zip(all_preds, all_knowledge))]), 4),
            "coverage": round(np.mean(coverages), 4),
            "density": round(np.mean(densities), 4),
            "lcs": round(np.mean(lcss), 4),
        }


        with open(self.out_results_file, "w") as f:
            json.dump(
                scores,
                f
            )