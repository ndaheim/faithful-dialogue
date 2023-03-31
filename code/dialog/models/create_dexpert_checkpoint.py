import argparse

from transformers import AutoConfig
from dexperts import PretrainedDensityRatioMethodModel, DensityRatioMethodConfig


parser = argparse.ArgumentParser()
parser.add_argument("--dm_model_name_or_path", type=str)
parser.add_argument("--ilm_model_name_or_path", type=str)
parser.add_argument("--lm_model_name_or_path", type=str)
parser.add_argument("--ilm_scaling_factor", type=str)
parser.add_argument("--lm_scaling_factor", type=str)
parser.add_argument("--checkpoint_path", type=str)

args, additional_args = parser.parse_known_args()

lm_config = AutoConfig.from_pretrained(args.lm_model_name_or_path)
config = DensityRatioMethodConfig(
    direct_model_tokenizer_name_or_path=args.dm_model_name_or_path,
    direct_model_name_or_path=args.dm_model_name_or_path,
    internal_language_model_tokenizer_name_or_path=args.ilm_model_name_or_path,
    internal_language_model_name_or_path=args.ilm_model_name_or_path,
    language_model_tokenizer_name_or_path=args.lm_model_name_or_path,
    language_model_name_or_path=args.lm_model_name_or_path,
    ilm_scaling_factor=float(args.ilm_scaling_factor),
    lm_scaling_factor=float(args.lm_scaling_factor),
    **lm_config.to_diff_dict()
)

model = PretrainedDensityRatioMethodModel(config=config)
model.save_pretrained(args.checkpoint_path)