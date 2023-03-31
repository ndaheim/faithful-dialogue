# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""
DSTC9 Track 1 - Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access - Dataset
"""

from __future__ import absolute_import, division, print_function

import json
from typing import List, Optional

import datasets
import numpy as np
from tqdm import tqdm

from .base import DocumentGroundedDataset


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{kim2020domain,
  title={Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access},
  author={Seokhwan Kim and Mihail Eric and Karthik Gopalakrishnan and Behnam Hedayatnia and Yang Liu and Dilek Hakkani-Tur},
  journal={arXiv preprint arXiv:2006.03533}
  year={2020}
}
"""

_DESCRIPTION = """\

"""

_HOMEPAGE = "https://github.com/alexa/alexa-with-dstc9-track1-dataset"


_BASE_URL = "https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master"
_URLs = {
    'train': {
        'logs': f'{_BASE_URL}/data/train/logs.json',
        'labels': f'{_BASE_URL}/data/train/labels.json',
        'knowledge': f'{_BASE_URL}/data/knowledge.json',
    },
    'val': {
        'logs': f'{_BASE_URL}/data/val/logs.json',
        'labels': f'{_BASE_URL}/data/val/labels.json',
        'knowledge': f'{_BASE_URL}/data/knowledge.json',
    },
    'test': {
        'logs': f'{_BASE_URL}/data_eval/test/logs.json',
        'labels': f'{_BASE_URL}/data_eval/test/labels.json',
        'knowledge': f'{_BASE_URL}/data_eval/knowledge.json',
    }
}

class DSTC9Track1(DocumentGroundedDataset, datasets.GeneratorBasedBuilder):

    def _get_knowledge_documents(self, labels, knowledge):
        docs = []

        for knowledge_label in labels["knowledge"]:
            domain, entity_id, doc_id = knowledge_label["domain"], str(knowledge_label["entity_id"]), str(knowledge_label["doc_id"])
            snippet = knowledge[domain][entity_id]["docs"][doc_id]
            docs.append({
                "description": "",
                "text": f"Q: {snippet['title']}, A: {snippet['body']}"
            })

        return docs

    def _get_user(self, turn):
        if "S" in turn["speaker"]:
            return "system"
        else:
            return "user"

    def _map_to_common_format(self, dialog, labels, knowledge):
        if not labels["target"]:
            return []
        
        context = []
        samples = []

        for turn in dialog:
            formatted_turn = {}

            # not annotated in DSTC9
            formatted_turn["dialog_act"] = ""
            formatted_turn["text"] = turn["text"]
            formatted_turn["user"] = self._get_user(turn)

            context.append(formatted_turn)

        knowledge_documents = self._get_knowledge_documents(labels, knowledge)

        if len(knowledge_documents) > 0:
            new_sample = {
                "context": context,
                "dataset_id": "DSTC9",
                "dialog_act": "",
                "knowledge": knowledge_documents,
                "response": labels["response"]
            }
            samples.append(new_sample)

        return samples

    def _split_generators(self, dl_manager):
        data = {}
        for key, value in _URLs.items():
            data[key] = {}
            for key_, url in value.items():
                url_to_download = url
                file_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
                with open(file_path, "r") as f:
                    data[key][key_] = json.load(f)

        splits = ["train", "val", "test"]

        formatted_data = {}

        for split in splits:
            formatted_data[split] = [self._map_to_common_format(dialog, labels, data[split]["knowledge"]) 
                                        for dialog, labels in tqdm(zip(data[split]["logs"], data[split]["labels"]))]
            formatted_data[split] = [sample for sample in formatted_data[split] if sample != []]

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": formatted_data[split],
                })
            for ds_split, split in
            zip([datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST], splits)
        ]
