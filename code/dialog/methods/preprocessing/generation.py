import itertools

from dialog.methods.preprocessing.base import Preprocessor

class Seq2SeqPreprocessor(Preprocessor):

    def preprocess(self, features):
        sequences, labels = [], []
        for source, target in zip(features["input"], features["output"]):
            tokenized_source = self.tokenizer(
                source, 
                max_length=self.model_args.max_input_length,
                truncation=True
            )["input_ids"]
            tokenized_target = self.tokenizer(
                target, 
                max_length=self.model_args.max_output_length,
                truncation=True
            )["input_ids"]
            sequences.append(tokenized_source)
            labels.append(tokenized_target)

        return sequences, labels


class DocumentGroundedPreprocessor(Preprocessor):

    def _process_knowledge(self, knowledge):
        knowledge = self.tokenizer(
            self.model_args.knowledge_sep_token.join([f"{k['text']}" for k in knowledge]),
            add_special_tokens=False
        )["input_ids"]
        knowledge = self._truncate_knowledge(knowledge)
        return knowledge

    def _truncate_knowledge(self, knowledge):
        if self.model_args.knowledge_truncation_strategy == "right":
            knowledge = knowledge[:self.model_args.knowledge_max_tokens]
        return knowledge

    def preprocess(self, features):
        sequences, labels = [], []
        for context, dialog_act, knowledge, response in zip(
                features["context"],
                features["dialog_act"],
                features["knowledge"],
                features["response"]
        ):
            context = self._process_dialog_context(context)
            response = self._process_response(response)
            knowledge = self._process_knowledge(knowledge)
            dialog_act = self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"]

            bos_token_needed = self.tokenizer.bos_token is not None
            full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

            full_sequence += [
                dialog_act,
                [self.tokenizer.convert_tokens_to_ids(self.model_args.knowledge_tag_token)],
                knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence = list(itertools.chain.from_iterable(full_sequence))

            sequences.append(full_sequence)
            labels.append(response)

        return sequences, labels

class DocumentGroundedPreprocessorWithKnowledgeLabels(DocumentGroundedPreprocessor):

    def preprocess(self, features):
        knowledge_labels = []
        sequences, labels = super().preprocess(features)

        for sample in features["knowledge"]:
            knowledge_labels.append(self.tokenizer(sample[0]["text"])["input_ids"])

        return sequences, labels, knowledge_labels

class DocumentGroundedPreprocessorWithCTRLTokens(DocumentGroundedPreprocessor):

    def _process_ctrl_tokens(self, ctrl_tokens):
        ctrl_tokens = self.tokenizer(
            ctrl_tokens,
            add_special_tokens=False
        )["input_ids"]
        ctrl_tokens = self._truncate_knowledge(ctrl_tokens)
        return ctrl_tokens

    def preprocess(self, features):
        sequences, labels = [], []
        for context, dialog_act, knowledge, ctrl_tokens, response in zip(
                features["context"],
                features["dialog_act"],
                features["knowledge"],
                features["ctrl_tokens"],
                features["response"]
        ):
            context = self._process_dialog_context(context)
            response = self._process_response(response)
            knowledge = self._process_knowledge(knowledge)
            if not self.data_args.is_training:
                ctrl_tokens = "<high-prec> <entailed>"
            ctrl_tokens = self._process_ctrl_tokens(ctrl_tokens)
            dialog_act = self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"]

            bos_token_needed = self.tokenizer.bos_token is not None
            full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

            full_sequence += [
                dialog_act,
                ctrl_tokens,
                knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence = list(itertools.chain.from_iterable(full_sequence))

            sequences.append(full_sequence)
            labels.append(response)

        return sequences, labels

class DocumentGroundedPreprocessorForDensityRatio(DocumentGroundedPreprocessorWithCTRLTokens):

    def preprocess(self, features):
        sequences, sequences_ctrl, labels = [], [], []
        for idx, (context, dialog_act, knowledge, response) in enumerate(zip(
                features["context"],
                features["dialog_act"],
                features["knowledge"],
                features["response"]
        )):
            context = self._process_dialog_context(context)
            response = self._process_response(response)
            knowledge = self._process_knowledge(knowledge)
            if "ctrl_tokens" in features:
                if not self.data_args.is_training:
                    ctrl_tokens = "<high-prec> <entailed>"
                ctrl_tokens = self._process_ctrl_tokens(ctrl_tokens)
            else: 
                ctrl_tokens = []
            dialog_act = self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"]

            bos_token_needed = self.tokenizer.bos_token is not None
            full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []
            full_sequence_ctrl = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

            full_sequence += [
                dialog_act,
                knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence_ctrl += [
                dialog_act,
                ctrl_tokens,
                knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence = list(itertools.chain.from_iterable(full_sequence))
            full_sequence_ctrl = list(itertools.chain.from_iterable(full_sequence_ctrl))

            sequences.append(full_sequence)
            sequences_ctrl.append(full_sequence_ctrl)
            labels.append(response)

        return sequences, sequences_ctrl, labels

class DocumentGroundedPreprocessorForNoisyChannel(DocumentGroundedPreprocessor):

    def preprocess(self, features):
        sequences, sequences_lm, labels = [], [], []
        for idx, (context, dialog_act, knowledge, response) in enumerate(zip(
                features["context"],
                features["dialog_act"],
                features["knowledge"],
                features["response"]
        )):
            context = self._process_dialog_context(context)
            response = self._process_response(response)
            knowledge = self._process_knowledge(knowledge)
            no_knowledge = self._process_knowledge([])
            dialog_act = self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"]

            bos_token_needed = self.tokenizer.bos_token is not None
            full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []
            full_sequence_lm = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

            full_sequence += [
                dialog_act,
                knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence_lm += [
                dialog_act,
                no_knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence = list(itertools.chain.from_iterable(full_sequence))
            full_sequence_lm = list(itertools.chain.from_iterable(full_sequence_lm))

            sequences.append(full_sequence)
            sequences_lm.append(full_sequence_lm)
            labels.append(response)

        return sequences, sequences_lm, labels