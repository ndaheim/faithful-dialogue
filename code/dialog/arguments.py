from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    method: str = field(default=384)

    # Seq2Seq model specific args
    generation_max_len: int = field(default=60)
    generation_beam_size: int = field(default=10)
    generation_do_sample: bool = field(default=False)
    generation_length_penalty: float = field(default=1.0)
    generation_uid_regularization: float = field(default=0.0)
    generation_no_repeat_ngram_size: int = field(default=3)
    num_return_sequences: int = field(default=1)
    num_sequences_to_keep: int = field(default=1)
    num_labels: int = field(default=None)

    # Tokenization
    history_max_tokens: int = field(default=384)
    history_max_utterances: int = field(default=999)
    knowledge_max_tokens: int = field(default=128)
    knowledge_truncation_strategy: str = field(default="right")
    max_input_length: int = field(default=1024)
    max_output_length: int = field(default=1024)

    # Special Tokens
    user_token: str = field(default="<user>")
    agent_token: str = field(default="<system>")
    system_token: str = field(default="<system>")
    knowledge_sep_token: str = field(default="<knowledge_sep>")
    knowledge_tag_token: str = field(default="<knowledge_tag>")

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_data_files: Optional[dict] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_transformations: Optional[List[str]] = field(
        default=None,
    )
    dataset_lowercase_entities: bool = field(default=False)
    dataset_filter_dict: Optional[dict] = field(
        default=None,
    )
    dataset_train_split: str = field(default="train")
    dataset_val_split: Optional[str] = field(default=None)

    is_training: bool = field(default=True)
    track_fim: bool = field(default=False)


@dataclass
class DataPredictionArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_data_files: Optional[dict] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_filter_dict: Optional[dict] = field(
        default=None,
    )
    dataset_transformations: Optional[List[str]] = field(
        default=None,
    )
    dataset_lowercase_entities: bool = field(default=False)
    dataset_test_split: str = field(default="test")

    test_documents_faiss_index_path: str = field(default=None)

    metric_output_file: Optional[str] = field(default=None)

    prediction_output_file: Optional[str] = field(default=None)

    is_training: bool = field(default=False)
