import argparse
from typing import Tuple

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, \
    PreTrainedTokenizerBase, PreTrainedModel
import torch

from bert_modeling import ContrastiveBertConfig, \
    ContrastiveBertForSequenceClassification
from trainer import train, sequence_classification_preprocess_dataset


def get_model_and_tokenizer(model_name_or_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    config = ContrastiveBertConfig.from_pretrained(model_name_or_path)
    config.num_labels = 2
    config.classifiers_layers = [12, 8, 4]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = ContrastiveBertForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    return model, tokenizer


def preprocess_sst_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    source_column = "sentence"
    target_column = "label"
    max_seq_length = 512
    return dataset.map(lambda examples: sequence_classification_preprocess_dataset(examples,
                                                                                   tokenizer,
                                                                                   source_column,
                                                                                   target_column,
                                                                                   max_seq_length,
                                                                                   max_seq_length),
                       remove_columns=dataset.column_names,
                       batched=True,
                       load_from_cache_file=False)


def train_script(output_dir: str):
    model, tokenizer = get_model_and_tokenizer("bert-base-cased")
    data_collator = DataCollatorWithPadding(tokenizer)

    train_dataset = load_dataset("sst2", split="train").select(range(10000))
    dev_dataset = load_dataset("sst2", split="validation")

    train_dataset = preprocess_sst_dataset(train_dataset, tokenizer)
    dev_dataset = preprocess_sst_dataset(dev_dataset, tokenizer)

    train(model, tokenizer, train_dataset, dev_dataset, data_collator, output_dir, 20)


def predict_script(model_name_or_path: str, output_dir: str):
    model, tokenizer = get_model_and_tokenizer(model_name_or_path)
    model.cuda()

    dev_dataset = load_dataset("sst2", split="validation")
    inputs = dev_dataset['sentence']
    labels = dev_dataset['label']

    logits_per_classifier = {l: torch.empty(0, model.config.num_labels, device="cuda")
                             for l in model.config.classifiers_layers}
    with torch.no_grad():
        batch_size = 8
        for i in range(0, len(inputs), batch_size):
            tokenized_inputs = tokenizer(inputs[i:min(i + batch_size, len(inputs))],
                                         padding=True,
                                         add_special_tokens=False,
                                         return_tensors="pt").to("cuda")
            batch_logits = model(**tokenized_inputs).logits
            for layer, l_logits in batch_logits.items():
                logits_per_classifier[layer] = torch.concat([logits_per_classifier[layer], l_logits])

    print("end")


def inference_script():
    inputs = tokenizer(
        "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_token_class_ids = logits.argmax(-1)

    # Note that tokens are classified rather then input words which means that
    # there might be more predicted token classes than words.
    # Multiple token classes might account for the same word
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
    print(predicted_tokens_classes)

    labels = predicted_token_class_ids
    loss = model(**inputs, labels=labels).loss
    round(loss.item(), 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    # region Train argparser
    parser_train = subparsers.add_parser('train', help='Train Agent Assist Summarizer')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-o', '--output-dir', required=True, type=str)
    # endregion

    # region Predict argparser
    parser_predict = subparsers.add_parser('predict', help='Train Agent Assist Summarizer')
    parser_predict.set_defaults(which='predict')
    parser_predict.add_argument('-o', '--output-dir', required=True, type=str)
    # endregion

    args = parser.parse_args()

    if args.which == "train":
        train_script(args.output_dir)
    if args.which == "predict":
        predict_script("temp/checkpoint-250", args.output_dir)
