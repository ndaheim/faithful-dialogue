import argparse
import json
import re
import string

import numpy as np
import spacy
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

nlp = spacy.load("en_core_web_sm")

"""
Re-implementation of Q^2 from: https://github.com/orhonovich/q-squared
"""

def get_answer_candidates(text):
    text = str(text)
    # print(text)
    doc = nlp(text)
    candidates = [ent.text for ent in list(doc.ents)]
    noun_chunks = list(doc.noun_chunks)
    for chunk in noun_chunks:
        found = False
        for cand in candidates:
            if chunk.text.lower() == cand.lower():
                found = True
        if not found:
            candidates.append(chunk.text)
    candidates = [cand for cand in candidates if cand.lower() != 'i']
    return candidates

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the|in|our)\b', ' ', text)
    return re.sub(' +', ' ', text).strip()

def calculate(predictions, dataset, num_questions, is_dstc=False):
    with torch.no_grad():
        qa_tokenizer = AutoTokenizer.from_pretrained("deepset/deberta-v3-large-squad2")#"ktrapeznikov/albert-xlarge-v2-squad-v2")
        qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-large-squad2").to("cuda:0")
        qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        qg_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap").to("cuda:0")
        f1 = 0
        num_questions = 0

        valid_questions = []
        valid_cands = []
        knowledge_answers = []
        scores = []

        for prediction, sample in tqdm(zip(predictions, dataset)):
            candidates = get_answer_candidates(prediction)
            if len(candidates) > 0:
                input_texts = []
                all_questions = []
                for cand in candidates:
                    input_texts.append(f"answer: {prediction} context: {cand} </s>")
                
                features = qg_tokenizer(input_texts, return_tensors='pt', padding=True).to("cuda:0")
                beam_outputs = qg_model.generate(**features, num_beams=4, num_return_sequences=1)

                questions = list(set([output.replace("<pad>", "").strip() for output in qg_tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)]))

                qa_inputs = qa_tokenizer(questions, [prediction]*len(questions), add_special_tokens=True, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
                for i, knowledge_sample in enumerate(sample["knowledge"]):
                    if is_dstc:
                        knowledge_sample = sample["knowledge"][i]["text"].split("A: ")[-1]
                    else:
                        knowledge_sample = sample["knowledge"][i]["text"]
                    qa_inputs_knowledge = qa_tokenizer(questions, [knowledge_sample]*len(questions), add_special_tokens=True, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
                    answer_start_scores, answer_end_scores = qa_model(**qa_inputs, return_dict=False)
                    answer_start_scores_knowledge, answer_end_scores_knowledge = qa_model(**qa_inputs_knowledge, return_dict=False)

                    answer_start = torch.argmax(answer_start_scores, dim=-1)
                    answer_end = torch.argmax(answer_end_scores, dim=-1) + 1

                    answer_start_knowledge = torch.argmax(answer_start_scores, dim=-1)
                    answer_end_knowledge = torch.argmax(answer_end_scores, dim=-1) + 1
                    
                    input_ids = qa_inputs["input_ids"].cpu()
                    input_ids_knowledge = qa_inputs_knowledge["input_ids"].cpu()


                    for i, (start, end) in enumerate(zip(answer_start.cpu().numpy(), answer_end.cpu().numpy())):
                        answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[i][start:end]))
                        answer_knowledge = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids_knowledge[i][start:end]))
                        if clean_text(answer) == clean_text(candidates[i]):
                            valid_questions.append(questions[i])
                            valid_cands.append(candidates[i])
                            knowledge_answers.append(answer_knowledge)

        del qa_model
        del qg_model
        # classifier = pipeline('zero-shot-classification', model='microsoft/deberta-v2-large-mnli')
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")

        for answer_knowledge, cand, question in tqdm(zip(
            knowledge_answers, valid_cands, valid_questions
        )):
            # premise = question + ' ' + answer_knowledge + '.'
            # hypothesis = question + ' ' + cand + '.'
            input_ = f"[CLS] {question} {answer_knowledge} [SEP] {question} {cand} [SEP]"
            inputs = tokenizer(input_, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
            outputs = model(**inputs)
            best = torch.argmax(outputs.logits)
            if best == 2:
                scores.append(1.0)
            elif best == 1:
                scores.append(0.5)
            else:
                scores.append(0.0)

        return np.mean(scores)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to a csv file containing dialogue model outputs.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True, help="Path to an output file")
    args = parser.parse_args()

    with open(args.infile) as f:
        predictions = json.load(f)

    dataset = load_dataset(args.dataset, "response_generation", split="test")

    num_questions = 10

    result = calculate(predictions, dataset, num_questions, is_dstc="dstc9" in args.dataset)
    
    with open(args.outfile, "w") as f:
        json.dump({
            "q2": result
        },
        f
    )