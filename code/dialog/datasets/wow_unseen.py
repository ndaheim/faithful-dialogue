import copy
import json
import os

import datasets
import numpy as np
from tqdm import tqdm

from .base import DocumentGroundedDataset


class WizardOfWikipedia(DocumentGroundedDataset, datasets.GeneratorBasedBuilder):
    _URL = "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz"

    def _get_knowledge_documents(self, turn):
        docs = []

        if "checked_sentence" in turn:
            if len(turn["checked_sentence"].values()) == 0:
                text = ""
            else:
                text = list(turn["checked_sentence"].values())[0]
            doc = {
                "description": "checked_sentence",
                "text": text
            }
            docs.append(doc)

        return docs

    def _get_user(self, turn):
        if "wizard" in turn["speaker"].lower():
            return "system"
        else:
            return "user"

    def _map_to_common_format(self, sample):
        context = []
        samples = []

        for turn in sample["dialog"]:
            formatted_turn = {}

            # not annotated in WoW
            formatted_turn["dialog_act"] = ""
            formatted_turn["text"] = turn["text"]
            formatted_turn["user"] = self._get_user(turn)

            knowledge_documents = self._get_knowledge_documents(turn)

            if len(knowledge_documents) > 0: #and knowledge_documents[0]["text"] != "no_passages_used":
                new_sample = {
                    "context": copy.deepcopy(context),
                    "dataset_id": "WizardOfWikipedia",
                    "dialog_act": "",
                    "knowledge": knowledge_documents,
                    "response": formatted_turn["text"]
                }
                samples.append(new_sample)

            context.append(formatted_turn)

        return samples

    def _split_generators(self, dl_manager):
        url_to_download = self._URL
        data_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        splits = ["train", "val", "test"]
        file_names = ["train.json", "valid_topic_split.json", "test_topic_split.json"]
        data_files = {
            split: os.path.join(data_path, file_name) for split, file_name in zip(splits, file_names)
        }

        formatted_data = {}

        for split in splits:
            with open(data_files[split], "r") as f:
                data = json.load(f)

            formatted_data[split] = [self._map_to_common_format(sample) for sample in tqdm(data)]

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": formatted_data[split]
                })
            for ds_split, split in
            zip([datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST], splits)
        ]
