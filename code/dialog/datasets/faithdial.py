import copy
import itertools
import json
import os
from typing import List

import datasets
import numpy as np
from datasets import load_dataset
from .base import DocumentGroundedDataset, DocumentGroundedDatasetConfig

_URLs = {
    "train": "https://huggingface.co/datasets/McGill-NLP/FaithDial/resolve/main/data/train.json",
    "validation": "https://huggingface.co/datasets/McGill-NLP/FaithDial/resolve/main/data/valid.json",
    "test": "https://huggingface.co/datasets/McGill-NLP/FaithDial/resolve/main/data/test.json"
}

class FaithDial(DocumentGroundedDataset, datasets.GeneratorBasedBuilder):

    def _get_context(self, turn):
        return [
            {
                "dialog_act": "",
                "text": turn,
                "user": "user" if i % 2 == 0 else "system"
            }
            for i, turn in enumerate(turn["history"])
        ]

    def _get_knowledge_documents(self, turn):
        docs = []

        doc = {
            "description": "",
            "text": turn["knowledge"]
        }
        docs.append(doc)

        return docs

    def _map_to_common_format(self, sample):
        samples = []

        for turn in sample["utterances"]:
            new_sample = {
                "context": self._get_context(turn),
                "dataset_id": "FaithDial",
                "dialog_act": "",
                "knowledge": self._get_knowledge_documents(turn),
                "response": turn["response"]
            }
            samples.append(new_sample)

        return samples

    def _map_to_common_format_for_hallucinations(self, sample):
        samples = []

        for turn in sample["utterances"]:
            if turn["response"] != turn["original_response"] and "Hallucination" in turn["BEGIN"]:
                response = turn["original_response"]
            else:
                continue
            new_sample = {
                "context": self._get_context(turn),
                "dataset_id": "FaithDial",
                "dialog_act": "",
                "knowledge": self._get_knowledge_documents(turn),
                "response": response
            }
            samples.append(new_sample)

        return samples


    def _map_to_ranking_format(self, sample):
        samples = []

        for turn in sample["utterances"]:
            if turn["original_response"] is not None:
                new_sample = {
                    "context": self._get_context(turn),
                    "dataset_id": "FaithDial",
                    "dialog_act": "",
                    "knowledge": self._get_knowledge_documents(turn),
                    "response": turn["response"],
                    "old_response": turn["original_response"]
                }
                samples.append(new_sample)

        return samples

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        splits = ["train", "validation", "test"]
        hf_splits = [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        data = {}
        
        for split in splits:
            url_to_download = _URLs[split] if self.config.name != "control" else _URLs_CTRL[split]
            data_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        
            with open(data_path, "r") as f:
                data[split] = json.load(f)

        if self.config.name in ["hallucinated_response", "hallucinated_response_ctrl"]:
            processing_function = self._map_to_common_format_for_hallucinations
        else:
            processing_function = self._map_to_common_format
        
        formatted_data = {}

        for split in splits:
            formatted_data[split] = [processing_function(dialog) for dialog in data[split]]
        
        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": formatted_data[split],
                })
            for ds_split, split in zip(hf_splits, splits)
        ]