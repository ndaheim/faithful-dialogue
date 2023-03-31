import itertools
from abc import ABC, abstractmethod


class Preprocessor(ABC):

    def __init__(self, config, data_args, model_args, tokenizer):
        self.config = config
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer

    def _process_dialog_context(self, context):
        context = [self._tokenize_with_special_tokens(turn) for turn in context]
        context = self._truncate_to_max_length(context)
        return list(itertools.chain.from_iterable(context))

    def _process_response(self, response):
        response = self.tokenizer(response)["input_ids"]
        if len(response) > 512:
            response = response[:511] + [response[-1]]
        return response

    def _tokenize_with_special_tokens(self, turn):
        dialog_act = self.tokenizer(turn["dialog_act"], add_special_tokens=False)["input_ids"]
        user_tag = self.model_args.user_token if turn["user"] == "user" else self.model_args.system_token
        user_tag = self.tokenizer(user_tag, add_special_tokens=False)["input_ids"]
        text = self.tokenizer(turn["text"], add_special_tokens=False)["input_ids"]
        return user_tag + dialog_act + text

    def _truncate_to_max_length(self, context):
        context_len = 0
        truncated_context = []

        context = context[-self.model_args.history_max_utterances:]
        for turn in context[::-1]:
            if context_len + len(turn) < self.model_args.history_max_tokens:
                truncated_context.append(turn)
                context_len += len(turn)
            else:
                break

        return truncated_context[::-1]

    @abstractmethod
    def preprocess(self, features):
        pass
