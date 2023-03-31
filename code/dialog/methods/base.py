import abc
from enum import Enum

from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import DataCollatorWithPadding, EvalPrediction, Seq2SeqTrainer


class Method(abc.ABC):
    def __init__(self, model_args, data_args, config, tokenizer):
        self.model_args = model_args
        self.data_args = data_args
        self.config = config
        self.tokenizer = tokenizer

        tokenizer.add_special_tokens({
            "additional_special_tokens": sorted(self.get_special_tokens())
            })
        self.metrics = []

    def get_special_tokens(self):
        return [
            self.model_args.user_token,
            self.model_args.system_token,
            self.model_args.knowledge_tag_token,
            self.model_args.knowledge_sep_token,
        ]

    @abc.abstractmethod
    def get_model_class(self, config):
        raise NotImplementedError()

    def get_model(self, run_mode, config):
        model_class = self.get_model_class(config)
        model = model_class.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=self.model_args.use_auth_token,
        )
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    @abc.abstractmethod
    def preprocess_features(self, features):
        raise NotImplementedError()

    def preprocess_features_and_maybe_normalize(self, features):
        # if self.data_args.dataset_transformations is not None:
        #     pipeline = Pipeline(self.data_args.dataset_transformations)
        #     for i, turns in enumerate(features["turns"]):
        #         for j, turn in enumerate(turns):
        #             features["turns"][i][j]["text"] = pipeline.apply(turns[j])
        return self.preprocess_features(features)

    def get_data_collator(self):
        return DataCollatorWithPadding(self.tokenizer)

    def get_trainer_class(self):
        return Trainer

    def postprocess_predictions(self, p, dataset):
        return p

    @abc.abstractmethod
    def compute_metrics(self, p: EvalPrediction):
        raise NotImplementedError()

    def _get_single_dataset(self, dataset, split, config_name):
        if "+" in config_name:
            config_names = config_name.split("+")
        else:
            config_names = [config_name]

        all_datasets = []

        for config_name in config_names:
            all_datasets.append(load_dataset(
                    dataset,
                    config_name,
                    split=split,
                    cache_dir=self.model_args.cache_dir,
                    data_files=self.data_args.dataset_data_files,
                    # dataset_filter_dict=self.data_args.dataset_filter_dict
                )
            )
        return concatenate_datasets(all_datasets)

    def _get_dataset(self, split, config_name=None):
        if config_name is None:
            config_name = self.data_args.dataset_config_name

        if isinstance(self.data_args.dataset_name, list):
            all_datasets = []
            for dataset in self.data_args.dataset_name:
                all_datasets.append(self._get_single_dataset(dataset, split, config_name))
            dataset = concatenate_datasets(all_datasets)
        else:
            dataset = self._get_single_dataset(self.data_args.dataset_name, split, config_name)

        old_eval_column_names = dataset.column_names

        dataset = dataset.map(
            self.preprocess_features_and_maybe_normalize,
            batched=True,
            batch_size=5000,
            load_from_cache_file=False
            #remove_columns=old_eval_column_names,
            )
        return dataset

    def get_train_dataset(self):
        return self._get_dataset(self.data_args.dataset_train_split)

    def get_test_dataset(self):
        return self._get_dataset(self.data_args.dataset_test_split)

    def get_validation_dataset(self):
        return self._get_dataset(self.data_args.dataset_val_split)


class TaskType(Enum):
    GENERATION = 1
    CLASSIFICATION = 2