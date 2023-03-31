import json
import os

import datasets
import numpy as np
from tqdm import tqdm
from .base import DocumentGroundedDataset


class DSTC11(DocumentGroundedDataset, datasets.GeneratorBasedBuilder):
    _URL = "https://raw.githubusercontent.com/alexa/dstc11-track5/main/data/"

    def _get_knowledge_documents(self, label, knowledge):
        docs = []

        for doc in label["knowledge"]:
            if doc["doc_type"] == "review":
                docs.append({
                    "description": "review",
                    "text": knowledge[doc["domain"]][str(doc["entity_id"])]["reviews"][str(doc["doc_id"])]["sentences"][str(doc["sent_id"])]
                })
            elif doc["doc_type"] == "faq":
                doc = knowledge[doc["domain"]][str(doc["entity_id"])]["faqs"][str(doc["doc_id"])]
                doc_text = f"Q: {doc['question']} A: {doc['answer']}"
                docs.append({
                    "description": "faq",
                    "text": doc_text
                })

        return docs

    def _get_user(self, turn):
        if turn["speaker"] == "S":
            return "system"
        else:
            return "user"

    def _map_to_common_format(self, dialog, label, knowledge):
        context = []

        for turn in dialog:
            formatted_turn = {}

            # not annotated in DSTC11
            formatted_turn["dialog_act"] = ""
            formatted_turn["text"] = turn["text"]
            formatted_turn["user"] = self._get_user(turn)
            context.append(formatted_turn)

        knowledge_documents = self._get_knowledge_documents(label, knowledge)

        sample = {
            "context": context,
            "dataset_id": "DSTC11",
            "dialog_act": "",
            "knowledge": knowledge_documents,
            "response": label["response"]
        }

        return [sample]

    def _split_generators(self, dl_manager):
        splits = ["train", "val"]
        data = {split: {} for split in splits}
        for split in splits:
            for handle in ["labels", "logs"]:
                url_to_download = os.path.join(self._URL, split, f"{handle}.json")
                data_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
                with open(data_path, "r") as f:
                    data[split][handle] = json.load(f)
        
        url_to_download = os.path.join(self._URL, "knowledge.json")
        data_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        with open(data_path, "r") as f:
            data["knowledge"] = json.load(f)

        formatted_data = {}

        for split in splits:
            formatted_data[split] = [self._map_to_common_format(dialog, labels, data["knowledge"]) 
                                        for dialog, labels in tqdm(zip(data[split]["logs"], data[split]["labels"])) if labels["target"]]
            formatted_data[split] = [sample for sample in formatted_data[split] if sample != []]

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": formatted_data[split],
                })
            for ds_split, split in
            zip([datasets.Split.TRAIN, datasets.Split.VALIDATION], splits)
        ]