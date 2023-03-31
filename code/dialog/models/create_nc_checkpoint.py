import argparse

from transformers import AutoConfig
from noisy_channel import PretrainedNoisyChannelModel, NoisyChannelConfig


parser = argparse.ArgumentParser()
parser.add_argument("--dm_model_name_or_path", type=str)
parser.add_argument("--cm_model_name_or_path", type=str)
parser.add_argument("--lm_model_name_or_path", type=str)
parser.add_argument("--cm_scaling_factor", type=str)
parser.add_argument("--lm_scaling_factor", type=str)
parser.add_argument("--length_penalty", type=str)
parser.add_argument("--checkpoint_path", type=str)

args, additional_args = parser.parse_known_args()
lm_config = AutoConfig.from_pretrained(args.lm_model_name_or_path)
config = NoisyChannelConfig(
    direct_model_tokenizer_name_or_path=args.dm_model_name_or_path,
    direct_model_name_or_path=args.dm_model_name_or_path,
    channel_model_tokenizer_name_or_path=args.cm_model_name_or_path,
    channel_model_name_or_path=args.cm_model_name_or_path,
    language_model_tokenizer_name_or_path=args.lm_model_name_or_path,
    language_model_name_or_path=args.lm_model_name_or_path,
    cm_scaling_factor=float(args.cm_scaling_factor),
    lm_scaling_factor=float(args.lm_scaling_factor),
    length_penalty=float(args.length_penalty),
    **lm_config.to_diff_dict()
)

model = PretrainedNoisyChannelModel(config=config)
model.save_pretrained(args.checkpoint_path)