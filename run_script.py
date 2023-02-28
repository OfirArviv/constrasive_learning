import copy
import os
import sys

from transformers.trainer_utils import get_last_checkpoint

if os.path.exists('/dccstor'):
    os.environ['TRANSFORMERS_CACHE'] = '/dccstor/sum-datasets/users/ofir.arviv/transformers_cache'
    os.environ['HF_HOME'] = '/dccstor/sum-datasets/ofir.arviv/transformers_cache'
    os.environ['HF_DATASETS_CACHE'] = '/dccstor/sum-datasets/ofir.arviv/transformers_datasets_cache'

import argparse
import glob
import itertools
import json
from typing import Tuple, List, Dict, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from netcal.presentation import ReliabilityDiagram
from netcal.scaling import TemperatureScaling
import evaluate
import numpy as np
from datasets import load_dataset, Dataset
from netcal.metrics.confidence import ECE
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, \
    PreTrainedTokenizerBase, PreTrainedModel, set_seed, DataCollator
import torch
import math
from bert_modeling import ContrastiveBertConfig, ContrastiveBertForSequenceClassification
from trainer import train
import pandas as pd
from json_numpy import default, object_hook


# region Utils

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


# endregion

# region Dataset processing
def preprocess_basic_classification_example(examples,
                                            tokenizer: PreTrainedTokenizerBase,
                                            input_column: str,
                                            label_column: str,
                                            max_source_length: int):
    # remove pairs where at least one record is None
    inputs, labels = [], []
    for i in range(len(examples[input_column])):
        if examples[input_column][i] is not None and examples[label_column][i] is not None:
            inputs.append(examples[input_column][i])
            labels.append(examples[label_column][i])

    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

    # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    #   labels = tokenizer(labels, max_length=max_target_length, truncation=False)

    model_inputs["labels"] = labels
    return model_inputs


def preprocess_basic_classification_dataset(dataset: Dataset,
                                            tokenizer: PreTrainedTokenizerBase,
                                            source_column: str,
                                            label_column: str
                                            ) -> Dataset:
    max_seq_length = 512
    return dataset.map(lambda examples: preprocess_basic_classification_example(examples,
                                                                                tokenizer,
                                                                                source_column,
                                                                                label_column,
                                                                                max_seq_length),
                       remove_columns=dataset.column_names,
                       batched=True,
                       load_from_cache_file=False)


def preprocess_pair_classification_example(examples,
                                           tokenizer: PreTrainedTokenizerBase,
                                           input_column_1: str,
                                           input_column_2: str,
                                           label_column: str,
                                           max_source_length: int):
    # remove pairs where at least one record is None
    inputs, labels = [], []
    for i in range(len(examples[input_column_1])):
        input_1 = examples[input_column_1][i]
        input_2 = examples[input_column_2][i]
        label = examples[label_column][i]
        if input_1 is not None and input_2 is not None and label is not None:
            if label not in [0, 1, 2]:
                continue

            inputs.append(f'{input_1}{tokenizer.sep_token}{input_2}')
            labels.append(label)

    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

    model_inputs["labels"] = labels
    return model_inputs


def preprocess_pair_classification_dataset(dataset: Dataset,
                                           tokenizer: PreTrainedTokenizerBase,
                                           input_column_1: str,
                                           input_column_2: str,
                                           label_column: str, ) -> Dataset:
    max_seq_length = 512
    return dataset.map(lambda examples: preprocess_pair_classification_example(examples,
                                                                               tokenizer,
                                                                               input_column_1,
                                                                               input_column_2,
                                                                               label_column,
                                                                               max_seq_length),
                       remove_columns=dataset.column_names,
                       batched=True,
                       load_from_cache_file=False)


def preprocess_qa_classification_example(examples,
                                         tokenizer: PreTrainedTokenizerBase,
                                         question_column: str,
                                         context_column: str,
                                         label_column: str,
                                         max_source_length: int):
    # remove pairs where at least one record is None
    inputs, labels = [], []
    for i in range(len(examples[question_column])):
        question = examples[question_column][i]
        context = examples[context_column][i]
        label = examples[label_column][i]
        if question is not None and context is not None and label is not None:
            if label not in [0, 1, 2]:
                continue

            inputs.append(f'question: {question} context: {context}')
            labels.append(label)

    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

    model_inputs["labels"] = labels
    return model_inputs


def preprocess_qa_classification_dataset(dataset: Dataset,
                                         tokenizer: PreTrainedTokenizerBase,
                                         question_column: str,
                                         context_column: str,
                                         label_column: str,
                                         ) -> Dataset:
    max_seq_length = 512
    return dataset.map(lambda examples: preprocess_qa_classification_example(examples,
                                                                             tokenizer,
                                                                             question_column,
                                                                             context_column,
                                                                             label_column,
                                                                             max_seq_length),
                       remove_columns=dataset.column_names,
                       batched=True,
                       load_from_cache_file=False)


def preprocess_multi_label_classification_example(examples,
                                                  tokenizer: PreTrainedTokenizerBase,
                                                  input_column: str,
                                                  label_column: str,
                                                  max_source_length: int,
                                                  num_labels: int):
    # remove pairs where at least one record is None
    inputs, labels = [], []
    for i in range(len(examples[input_column])):
        if examples[input_column][i] is not None and examples[label_column][i] is not None:
            inputs.append(examples[input_column][i])
            labels.append(examples[label_column][i])

    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

    # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    #   labels = tokenizer(labels, max_length=max_target_length, truncation=False)
    batch_size = len(labels)
    formatted_labels = np.zeros([batch_size, num_labels])
    for i, labels_idxs in enumerate(labels):
        for idx in labels_idxs:
            formatted_labels[i][idx] = 1
    model_inputs["labels"] = formatted_labels
    return model_inputs


def preprocess_multi_label_classification_dataset(dataset: Dataset,
                                                  tokenizer: PreTrainedTokenizerBase,
                                                  source_column: str,
                                                  label_column: str,
                                                  num_labels: int
                                                  ) -> Dataset:
    max_seq_length = 512
    return dataset.map(lambda examples: preprocess_multi_label_classification_example(examples,
                                                                                      tokenizer,
                                                                                      source_column,
                                                                                      label_column,
                                                                                      max_seq_length,
                                                                                      num_labels),
                       remove_columns=dataset.column_names,
                       batched=True,
                       load_from_cache_file=False)


def get_processed_dataset(dataset_key: str, split: str, tokenizer: PreTrainedTokenizerBase,
                          size: Optional[int] = None) -> Dataset:
    if dataset_key == "sst2":
        dataset_specific_args = {
            "source_column": "sentence",
            "label_column": "label",
        }
        dataset_preprocess_func = preprocess_basic_classification_dataset
        dataset = load_dataset(dataset_key, split=split)
    elif dataset_key == "snli":
        dataset_specific_args = {
            "input_column_1": "premise",
            "input_column_2": "hypothesis",
            "label_column": "label",
        }
        dataset_preprocess_func = preprocess_pair_classification_dataset
        dataset = load_dataset(dataset_key, split=split)
    elif dataset_key == "boolq":
        dataset_specific_args = {
            "input_column_1": "question",
            "input_column_2": "passage",
            "label_column": "label",
        }
        dataset_preprocess_func = preprocess_pair_classification_dataset
        dataset = load_dataset("super_glue", name="boolq", split=split)
    elif dataset_key == "wnli":
        dataset_specific_args = {
            "input_column_1": "sentence1",
            "input_column_2": "sentence2",
            "label_column": "label",
        }
        dataset_preprocess_func = preprocess_pair_classification_dataset
        dataset = load_dataset("glue", name="wnli", split=split)
    elif dataset_key == "cola":
        dataset_specific_args = {
            "source_column": "sentence",
            "label_column": "label",
        }
        dataset_preprocess_func = preprocess_basic_classification_dataset
        dataset = load_dataset("glue", name="cola", split=split)
    elif dataset_key == "ax":
        dataset_specific_args = {
            "input_column_1": "premise",
            "input_column_2": "hypothesis",
            "label_column": "label",
        }
        dataset_preprocess_func = preprocess_pair_classification_dataset
        dataset = load_dataset("glue", name="ax", split=split)
    elif dataset_key == "mrpc":
        dataset_specific_args = {
            "input_column_1": "sentence1",
            "input_column_2": "sentence2",
            "label_column": "label",
        }
        dataset_preprocess_func = preprocess_pair_classification_dataset
        dataset = load_dataset("glue", name="mrpc", split=split)
    elif dataset_key == "qqp":
        dataset_specific_args = {
            "input_column_1": "question1",
            "input_column_2": "question2",
            "label_column": "label",
        }
        dataset_preprocess_func = preprocess_pair_classification_dataset
        dataset = load_dataset("glue", name="qqp", split=split)
    elif dataset_key in ["mnli", "mnli_matched", "mnli_mismatched"]:
        dataset_specific_args = {
            "input_column_1": "premise",
            "input_column_2": "hypothesis",
            "label_column": "label",
        }
        dataset_preprocess_func = preprocess_pair_classification_dataset
        if dataset_key == "mnli" and split != "train":
            split = f'{split}_matched'
        dataset = load_dataset("glue", name=dataset_key, split=split)
    elif dataset_key == "banking77":
        dataset_specific_args = {
            "source_column": "text",
            "label_column": "label",
        }
        if split == "validation":
            split = "test"
        dataset_preprocess_func = preprocess_basic_classification_dataset
        dataset = load_dataset(dataset_key, split=split)
    elif dataset_key == "clinc-oos":
        dataset_specific_args = {
            "source_column": "text",
            "label_column": "intent",
        }
        dataset_preprocess_func = preprocess_basic_classification_dataset
        dataset = load_dataset("clinc-oos", name='plus', split=split)
    elif dataset_key == "go-emotions":
        dataset_specific_args = {
            "source_column": "text",
            "label_column": "labels",
            "num_labels": 28
        }
        dataset_preprocess_func = preprocess_multi_label_classification_dataset
        dataset = load_dataset("go_emotions", name='simplified', split=split)
    elif "xnli" in dataset_key:
        lang = dataset_key.split("xnli_")[1]
        dataset_specific_args = {
            "input_column_1": "premise",
            "input_column_2": "hypothesis",
            "label_column": "label",
        }
        dataset_preprocess_func = preprocess_pair_classification_dataset
        dataset = load_dataset("xnli", name=lang, split=split)
    else:
        raise NotImplementedError(dataset_key)

    dataset = dataset.shuffle(seed=42)
    if size is not None:
        size = min(size, len(dataset))
        dataset = dataset.select(range(size))
    dataset = dataset_preprocess_func(dataset, tokenizer, **dataset_specific_args)

    return dataset


# endregion

# region Model Func

def get_score(labels: List[int], prediction_per_classifier: Dict[str, List[int]]) -> Dict:
    num_labels = len(set(labels))
    if num_labels == 2:
        fscore_average = "binary"
    else:
        fscore_average = "weighted"

    fscore = evaluate.load("f1")
    f1_per_classifier = {f'{l}_f1': fscore.compute(references=labels, predictions=prediction_per_classifier[l],
                                                   average=fscore_average)['f1']
                         for l in prediction_per_classifier}

    acc_score = evaluate.load("accuracy")
    acc_per_classifier = {
        f'{l}_acc': acc_score.compute(references=labels, predictions=prediction_per_classifier[l])['accuracy']
        for l in prediction_per_classifier}

    res_dict = f1_per_classifier
    res_dict.update(acc_per_classifier)

    return res_dict


def get_model(model_name_or_path: str,
              classifiers_layers: List[int] = None,
              num_labels: Optional[int] = None,
              share_classifiers_weights: Optional[bool] = None
              ) -> PreTrainedModel:
    config = ContrastiveBertConfig.from_pretrained(model_name_or_path)
    if config.classifiers_layers is None:
        assert share_classifiers_weights is not None, 'share_classifiers_weights is None. Config does not contain ' \
                                                      'param share_classifiers_weights and thus it needs to be provided'
        assert num_labels is not None
        config.num_labels = num_labels
        config.classifiers_layers = classifiers_layers
        config.share_classifiers_weights = share_classifiers_weights

    # TODO: For some reason this config isnt saved
    else:
        config.num_labels = num_labels
        config.problem_type = None

    model = ContrastiveBertForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    return model


def get_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def train_script(dataset_key: str, share_classifiers_weights: bool, output_dir: str, classifiers_layers: List[int],
                 train_max_size: Optional[int], model_name: str):
    tokenizer = get_tokenizer(model_name)

    train_dataset = get_processed_dataset(dataset_key, "train", tokenizer, train_max_size)
    dev_dataset = get_processed_dataset(dataset_key, "validation", tokenizer)

    labels = train_dataset['labels']
    sample_label = labels[0]
    if isinstance(sample_label, list):
        num_labels = len(sample_label)
    else:
        label_set = set(train_dataset['labels'])
        num_labels = len(label_set)

    model = get_model(model_name, classifiers_layers, num_labels, share_classifiers_weights)
    data_collator = DataCollatorWithPadding(tokenizer)

    use_cpu = use_cpu = torch.cuda.is_available()
    model.to("cpu" if use_cpu else "cuda")

    train(model, tokenizer, train_dataset, dev_dataset, data_collator, output_dir, 5, no_cuda=use_cpu)


def _save_model_outputs_to_cache(model_name_or_path: str, dataset_key: str, output_dir: str, split: str,
                                 max_examples: Optional[int], logits_per_classifier: Dict, labels: List) -> None:
    model_name = model_name_or_path.replace("\\", "_").replace("/", "_")
    output_path = f'{output_dir}/model_{model_name}_dataset_{dataset_key}_split_{split}'
    if max_examples is not None:
        output_path = f'{output_path}_max_examples_{max_examples}'
    output_path = f'{output_path}.json'
    os.makedirs(output_dir, exist_ok=True)

    logits_per_classifier = copy.deepcopy(logits_per_classifier)
    for k in logits_per_classifier.keys():
        logits = logits_per_classifier[k]
        logits = logits.tolist()
        logits_per_classifier[k] = logits

    output_dic = {
        "logits_per_classifier": logits_per_classifier,
        "labels": labels
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dic, f, indent=4)


def _get_model_outputs_from_cache(model_name_or_path: str, dataset_key: str, output_dir: str, split: str,
                                  max_examples: Optional[int]) -> Optional[Dict]:
    model_name = model_name_or_path.split("models/")[1].replace("\\", "/").split("/")[0]
    output_path = f'{output_dir}/model_{model_name}_dataset_{dataset_key}_split_{split}'
    if max_examples is not None:
        output_path = f'{output_path}_max_examples_{max_examples}'
    output_path = f'{output_path}.json'

    if not os.path.isfile(output_path):
        return None

    with open(output_path, 'r', encoding='utf-8') as f:
        output_dict = json.load(f)

    logits_per_classifier = output_dict['logits_per_classifier']
    formatted_logits_per_classifier = dict()
    for k in logits_per_classifier.keys():
        logits = logits_per_classifier[k]
        logits = torch.tensor(logits)
        formatted_logits_per_classifier[int(k)] = logits

    output_dict['logits_per_classifier'] = formatted_logits_per_classifier

    return output_dict


def predict_script(model_name_or_path: str, dataset_key: str, output_dir: str, split: str = "validation",
                   use_cpu: bool = True, use_cache: bool = True,
                   max_examples: Optional[int] = None) -> Tuple[Dict[int, torch.Tensor], List[int]]:
    if use_cache:
        cached_dict = _get_model_outputs_from_cache(model_name_or_path, dataset_key, output_dir, split, max_examples)
        if cached_dict is not None:
            logits_per_classifier = cached_dict['logits_per_classifier']
            labels = cached_dict['labels']

            return logits_per_classifier, labels

    model_name_or_path = get_last_checkpoint(model_name_or_path)

    tokenizer = get_tokenizer(model_name_or_path)

    dev_dataset = get_processed_dataset(dataset_key, split, tokenizer, max_examples)
    labels = dev_dataset['labels']
    sample_label = labels[0]
    if isinstance(sample_label, list):
        num_labels = len(sample_label)
    else:
        label_set = set(dev_dataset['labels'])
        num_labels = len(label_set)

    model = get_model(model_name_or_path, num_labels)
    model.to("cpu" if use_cpu else "cuda")

    data_collator: DataCollator = DataCollatorWithPadding(tokenizer)

    logits_per_classifier = {l: torch.empty(0, model.config.num_labels, device="cpu" if use_cpu else "cuda")
                             for l in model.config.classifiers_layers}
    with torch.no_grad():
        batch_size = 8
        for i in tqdm(range(0, len(dev_dataset), batch_size)):
            tokenized_inputs = data_collator(dev_dataset[i:min(i + batch_size, len(dev_dataset))]).to(
                "cpu" if use_cpu else "cuda")
            batch_logits = model(**tokenized_inputs).logits
            for layer, l_logits in batch_logits.items():
                logits_per_classifier[layer] = torch.concat([logits_per_classifier[layer], l_logits])

    _save_model_outputs_to_cache(model_name_or_path, dataset_key, output_dir,
                                 split, max_examples, logits_per_classifier, labels)

    return logits_per_classifier, labels


# endregion

# region Uncertainty Calibration

def _adjust_binary_confidences(confidences_per_classifier: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    is_binary_confidence = list(confidences_per_classifier.values())[0].shape[1] == 2
    if is_binary_confidence:
        binary_confidence_per_classifier = {l: confidences_per_classifier[l][:, 1]
                                            for l in confidences_per_classifier.keys()}
        return binary_confidence_per_classifier
    else:
        return confidences_per_classifier


def plot_reliability_diagram_per_layer(confidences_per_classifier: Dict[int, np.ndarray], labels: List[int],
                                       label_suffix: str = "") -> None:
    confidences_per_classifier = _adjust_binary_confidences(confidences_per_classifier)
    for l in confidences_per_classifier.keys():
        confidences = confidences_per_classifier[l]
        ground_truth = np.asarray(labels)

        diag = ReliabilityDiagram(bins=10)
        diag.plot(confidences, ground_truth, title_suffix=f'{label_suffix}layer_{l}')
        plt.show()


def get_calibration_scores_per_layer(confidences_per_classifier: Dict[int, np.ndarray], labels: List[int]) -> Dict:
    confidences_per_classifier = _adjust_binary_confidences(confidences_per_classifier)
    score_dict = dict()
    for l in confidences_per_classifier.keys():
        confidences = confidences_per_classifier[l]
        ground_truth = np.asarray(labels)

        n_bins = 20
        ece = ECE(n_bins)
        ece_score = ece.measure(confidences, ground_truth)

        score_dict[l] = {f'ECE score': f'{ece_score}'}
    return score_dict


def temperature_calibration(confidences_per_classifier: Dict[int, np.ndarray],
                            labels: List[int]) -> Dict[int, np.ndarray]:
    calibrated_confidences_per_classifier = {}
    for l in confidences_per_classifier.keys():
        confidences = confidences_per_classifier[l]
        ground_truth = np.asarray(labels)
        temperature = TemperatureScaling()
        temperature.fit(confidences, ground_truth)
        calibrated = temperature.transform(confidences)
        calibrated_confidences_per_classifier[l] = calibrated

    # if binary logit transform from single logit format to multi-logit format
    is_binary_confidence = list(confidences_per_classifier.values())[0].shape[1] == 2
    if is_binary_confidence:
        calibrated_confidences_per_classifier = {k: np.array(list(map(lambda x: [1 - x, x], v)))
                                                 for k, v in calibrated_confidences_per_classifier.items()}

    return calibrated_confidences_per_classifier


# endregion

# region Contrastive Utils

def get_lower_layer_overconfidence(higher_layer_confidences: np.ndarray, lower_layer_confidences: np.ndarray):
    # We check the case where both layers agree. This seems ti be the 'classic case' for contrastive learning.
    #  There are other cases, for example, where the lower layer thinks differently from the higher layer,
    #  TODO: maybe we can do something there too?
    higher_layer_prob_argmax = np.argmax(higher_layer_confidences)
    lower_layer_prob = lower_layer_confidences[higher_layer_prob_argmax]
    higher_layer_prob = higher_layer_confidences[higher_layer_prob_argmax]

    if lower_layer_prob > higher_layer_prob:
        lower_layer_overconfidence = ((lower_layer_prob / higher_layer_prob) - 1)
    else:
        lower_layer_overconfidence = 0

    return lower_layer_overconfidence


def get_contrastive_predictions(confidences_per_classifier: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
    num_labels = list(confidences_per_classifier.values())[0].shape[1]
    contrastive_confidences_dict = dict()
    for layer_1, layer_2 in itertools.combinations(confidences_per_classifier.keys(), 2):
        higher_layer = layer_1 if layer_1 > layer_2 else layer_2
        lower_layer = layer_1 if higher_layer == layer_2 else layer_2

        higher_layer_confidence = confidences_per_classifier[higher_layer]
        lower_layer_confidence = confidences_per_classifier[lower_layer]

        lower_level_overconfidences = np.array(list(map(lambda x: get_lower_layer_overconfidence(x[0], x[1]),
                                                        zip(higher_layer_confidence, lower_layer_confidence))))
        lower_level_overconfidences = np.array(list(map(lambda x: [x] * num_labels, lower_level_overconfidences)))

        contrastive_confidences = higher_layer_confidence - lower_level_overconfidences * lower_layer_confidence
        contrastive_confidences_dict[f'{higher_layer}-{lower_layer}'] = contrastive_confidences

    contrastive_predictions = {l: np.argmax(contrastive_confidences_dict[l], axis=1).tolist()
                               for l in contrastive_confidences_dict}

    return contrastive_predictions


def get_layers_predictions_comparison(confidences_per_classifier: Dict[int, np.ndarray],
                                      labels: List[int],
                                      overconfidence_threshold: float,
                                      higher_layer_confidence_threshold: float) -> Tuple[Dict, Dict]:
    data_overconfidence = dict()
    data_higher_layer_confidence = dict()
    for layer_1, layer_2 in itertools.combinations(confidences_per_classifier.keys(), 2):
        higher_layer = layer_1 if layer_1 > layer_2 else layer_2
        lower_layer = layer_1 if higher_layer == layer_2 else layer_2

        data_overconfidence[f'{higher_layer}_{lower_layer}'] = {
            f'{higher_layer} correct': list(),
            f'{lower_layer} correct': list(),
            f'both correct': list(),
            f'both incorrect': list(),
            'all': list(),
        }

        data_higher_layer_confidence[f'{higher_layer}_{lower_layer}'] = {
            f'{higher_layer} correct': list(),
            f'{lower_layer} correct': list(),
            f'both correct': list(),
            f'both incorrect': list(),
            f'all': list()
        }

        curr_overconfidence_data = data_overconfidence[f'{higher_layer}_{lower_layer}']
        curr_higher_layer_confidence_data = data_higher_layer_confidence[f'{higher_layer}_{lower_layer}']

        higher_layer_confidences = confidences_per_classifier[higher_layer]
        lower_layer_confidences = confidences_per_classifier[lower_layer]
        overconfidences_list = []
        for i in range(len(higher_layer_confidences)):
            higher_layer_confidence = higher_layer_confidences[i]
            lower_layer_confidence = lower_layer_confidences[i]

            lower_layer_overconfidence = get_lower_layer_overconfidence(higher_layer_confidence,
                                                                        lower_layer_confidence)

            overconfidences_list.append(lower_layer_overconfidence)
            higher_layer_prediction = np.argmax(higher_layer_confidence)
            higher_layer_prediction_confidence = np.max(higher_layer_confidence)
            lower_layer_prediction = np.argmax(lower_layer_confidence)

            curr_overconfidence_data[f'all'].append(lower_layer_overconfidence)
            curr_higher_layer_confidence_data[f'all'].append(higher_layer_prediction_confidence)
            if lower_layer_overconfidence > overconfidence_threshold and higher_layer_prediction_confidence > higher_layer_confidence_threshold:
                label = labels[i]

                if label == higher_layer_prediction and label == lower_layer_prediction:
                    axis_key = "both correct"
                elif label == higher_layer_prediction:
                    axis_key = f'{higher_layer} correct'
                elif label == lower_layer_prediction:
                    axis_key = f'{lower_layer} correct'
                else:
                    axis_key = f'both incorrect'

                curr_overconfidence_data[axis_key].append(lower_layer_overconfidence)
                curr_higher_layer_confidence_data[axis_key].append(higher_layer_prediction_confidence)

    return data_overconfidence, data_higher_layer_confidence


# endregion

# region Experiments

def get_optimal_contrastive_thresholds(confidences_per_classifier: Dict[int, np.ndarray],
                                       labels: List[int],
                                       layers_combinations: List[str]) -> Dict:
    layer_comparison_dict = get_layer_comparison_df(confidences_per_classifier, labels, layers_combinations)
    res_dict = dict()
    for layers_selection in layers_combinations:
        df_both_correct_cnt = layer_comparison_dict[layers_selection]["both correct"]
        df_both_incorrect_cnt = layer_comparison_dict[layers_selection]["both incorrect"]

        diff_df = df_both_incorrect_cnt - df_both_correct_cnt

        max_val = -sys.maxsize
        optimal_max_row_idx = optimal_max_col_idx = optimal_min_row_idx = optimal_min_col_idx = None
        for max_row_idx, max_col_idx in itertools.product(diff_df.index, diff_df.columns):
            for min_row_idx, min_col_idx in itertools.product(diff_df.index, diff_df.columns):
                if max_row_idx > min_row_idx and max_col_idx > min_col_idx:
                    val = diff_df.loc[min_row_idx:max_row_idx, min_col_idx:max_col_idx].sum().sum()
                    if val > max_val:
                        max_val = val
                        optimal_min_col_idx = min_col_idx
                        optimal_min_row_idx = min_row_idx
                        optimal_max_col_idx = max_col_idx
                        optimal_max_row_idx = max_row_idx
        res_dict[layers_selection] = {
            "max val": max_val,
            "optimal_max_col_idx": optimal_max_col_idx,
            "optimal_max_row_idx": optimal_max_row_idx,
            "optimal_min_col_idx": optimal_min_col_idx,
            "optimal_min_row_idx": optimal_min_row_idx

        }

    return res_dict


def get_native_oracle_contrastive_score(confidences_per_classifier: Dict[int, np.ndarray],
                                        labels: List[int]) -> Dict:
    data_overconfidence, data_higher_layer_confidence = \
        get_layers_predictions_comparison(confidences_per_classifier, labels, 0, 0)
    thresholds_dict = get_optimal_contrastive_thresholds(confidences_per_classifier, labels,
                                                         list(data_overconfidence.keys()))

    prediction_per_classifier = {l: np.argmax(confidences_per_classifier[l], axis=1).tolist()
                                 for l in confidences_per_classifier}
    optimal_prediction_per_classifier = dict()
    for l in data_overconfidence.keys():
        higher_layer = int(l.split("_")[0])
        optimal_prediction = prediction_per_classifier[higher_layer].copy()
        thresh = thresholds_dict[l]
        cnt = 0
        for i in range(len(optimal_prediction)):
            overconfidence = data_overconfidence[l]['all'][i]
            higher_layer_confidence = data_higher_layer_confidence[l]['all'][i]
            if overconfidence == 0:
                continue
            if (thresh['optimal_min_row_idx'] <= overconfidence < thresh['optimal_max_row_idx']) and (
                    thresh['optimal_min_col_idx'] <= higher_layer_confidence < thresh['optimal_max_col_idx']):
                cnt += 1
                optimal_prediction[i] = labels[i]
        optimal_prediction_per_classifier[l] = optimal_prediction

    scores = get_score(labels, prediction_per_classifier)
    optimal_scores = get_score(labels, optimal_prediction_per_classifier)

    res = {
        "scores": scores,
        "optimal_scores": optimal_scores
    }

    return res


def get_layer_comparison_df(confidences_per_classifier: Dict[int, np.ndarray],
                            labels: List[int],
                            layers_combinations: List[str]) -> Dict:
    y_bins = list(np.round(np.linspace(0, 0.9, 10), 2))
    x_bins = list(np.round(np.linspace(0, 0.9, 10), 2))

    # X-axis - overconfidence
    # Y-axis - upper layer confidence
    # Z-axis - (None correct - both correct)
    df = pd.DataFrame(columns=y_bins, index=x_bins)
    df = df.rename_axis("overconfidence", axis=0)
    df = df.rename_axis("higher layer confidence", axis=1)
    df = df.fillna(0)

    data_overconfidence, data_higher_layer_confidence = \
        get_layers_predictions_comparison(confidences_per_classifier, labels, 0, 0)

    output_dict = dict()
    for i, layers_selection in enumerate(layers_combinations):
        df_both_correct_cnt = df.copy(deep=True)
        df_both_incorrect_cnt = df.copy(deep=True)

        for prediction_axis in ["both correct", "both incorrect"]:
            selected_overconfidence_data = data_overconfidence[layers_selection][prediction_axis]
            selected_higher_layer_confidence_data = data_higher_layer_confidence[layers_selection][prediction_axis]
            for higher_layer_confidence, lower_layer_overconfidence in zip(selected_higher_layer_confidence_data,
                                                                           selected_overconfidence_data):
                higher_layer_confidence_bin = int(math.floor(higher_layer_confidence * 100) / 10) * 10 / 100.0
                lower_layer_overconfidence_bin = int(math.floor(lower_layer_overconfidence * 100) / 10) * 10 / 100.0
                lower_layer_overconfidence_bin = min(lower_layer_overconfidence_bin, 0.9)
                if prediction_axis == "both correct":
                    df_both_correct_cnt.loc[lower_layer_overconfidence_bin, higher_layer_confidence_bin] += 1
                else:
                    df_both_incorrect_cnt.loc[lower_layer_overconfidence_bin, higher_layer_confidence_bin] += 1
        output_dict[layers_selection] = {
            "both correct": df_both_correct_cnt,
            "both incorrect": df_both_incorrect_cnt
        }

    return output_dict


def plot_layer_comparison_heatmap(confidences_per_classifier: Dict[int, np.ndarray],
                                  labels: List[int],
                                  title_prefix: str,
                                  layers_combinations: List[str]):
    layer_comparison_dict = get_layer_comparison_df(confidences_per_classifier, labels, layers_combinations)

    fig, axs = plt.subplots(ncols=len(layers_combinations), figsize=(7*len(layers_combinations), 7))
    for i, layers_selection in enumerate(layers_combinations):
        df_both_correct_cnt = layer_comparison_dict[layers_selection]["both correct"]
        df_both_incorrect_cnt = layer_comparison_dict[layers_selection]["both incorrect"]

        annotated_df = pd.DataFrame()
        diff_df = df_both_incorrect_cnt - df_both_correct_cnt
        for col, row in itertools.product(df_both_correct_cnt.columns, df_both_correct_cnt.index):
            both_correct_cnt = df_both_correct_cnt.loc[row, col]
            both_incorrect_cnt = df_both_incorrect_cnt.loc[row, col]
            diff_cnt = both_incorrect_cnt - both_correct_cnt

            diff_cnt_formatted = human_format(diff_cnt)
            both_correct_cnt_formatted = human_format(both_correct_cnt)
            both_incorrect_cnt_formatted = human_format(both_incorrect_cnt)
            annt = f'{diff_cnt_formatted} \n ({both_incorrect_cnt_formatted},\n{both_correct_cnt_formatted})'
            annotated_df.loc[row, col] = annt

        ax = sns.heatmap(diff_df, annot=annotated_df, linewidth=0.5, center=0, xticklabels=True, yticklabels=True,
                         vmax=3, vmin=-3, ax=axs[i], cbar=False, fmt='')
        ax.set(title=f'{title_prefix} - Layers {layers_selection}: None Correct - Both Correct')
    plt.tight_layout()
    plt.show()


# endregion

# region Batch scripts

def plot_layer_comparison_heatmap_for_all_models(split: str = "validation"):
    for model_dir in glob.glob("models/*"):
        dataset_key = os.path.basename(model_dir).split("_")[0]
        if dataset_key in ["boolq"]:
            continue
        model_path = f'{model_dir}/data/'
        print(dataset_key)
        logits_per_classifier, labels = predict_script(model_path, dataset_key, split=split)
        print(len(labels))
        confidence_per_classifier = {l: torch.nn.Softmax(dim=1)(logits_per_classifier[l]).cpu().numpy()
                                     for l in logits_per_classifier.keys()}
        calibrated_confidence_per_classifier = temperature_calibration(confidence_per_classifier, labels)
        plot_layer_comparison_heatmap(calibrated_confidence_per_classifier, labels, f'{dataset_key}: {split}')


def batch_run(split: str = "validation", output_dir: str = "models_output"):
    for model_dir in glob.glob("models/*"):
        dataset_key = os.path.basename(model_dir).split("_")[0]
        if dataset_key not in ["boolq", "go_emotions"]:
            continue
        dataset_key = "go_emotions"
        model_path = f'{model_dir}/data/'
        model_path = f'{"models/go_emotions_not_shared_bert"}/data/'
        logits_per_classifier, labels = predict_script(model_path, dataset_key, output_dir, split=split)
        confidence_per_classifier = {l: torch.nn.Softmax(dim=1)(logits_per_classifier[l]).cpu().numpy()
                                     for l in logits_per_classifier.keys()}
        calibrated_confidence_per_classifier = temperature_calibration(confidence_per_classifier, labels)
        get_native_oracle_contrastive_score(calibrated_confidence_per_classifier, labels)


# endregion

def plot_overconfidence_histogram(confidences_per_classifier: Dict[int, np.ndarray], labels: List[int]):
    for layer_1, layer_2 in itertools.combinations(confidences_per_classifier.keys(), 2):
        higher_layer = layer_1 if layer_1 > layer_2 else layer_2
        lower_layer = layer_1 if higher_layer == layer_2 else layer_2

        higher_layer_confidences = confidences_per_classifier[higher_layer]
        lower_layer_confidences = confidences_per_classifier[lower_layer]
        overconfidences_list = []
        for i in range(len(higher_layer_confidences)):
            higher_layer_confidence = higher_layer_confidences[i]
            lower_layer_confidence = lower_layer_confidences[i]

            higher_layer_prediction = np.argmax(higher_layer_confidence)
            lower_layer_prediction = np.argmax(lower_layer_confidence)
            label = labels[i]

            if higher_layer_prediction == label or lower_layer_prediction == label:
                continue

            lower_layer_overconfidence = get_lower_layer_overconfidence(higher_layer_confidence,
                                                                        lower_layer_confidence)
            overconfidences_list.append(lower_layer_overconfidence)

        plt.hist(overconfidences_list)
        plt.title(f'{layer_1}-{layer_2}-both wrong')
        plt.show()
    return None


def print_contrastive_results_comparison(model_name_or_path: str, dataset_key: str):
    logits_per_classifier, labels = predict_script(model_name_or_path, dataset_key)

    prediction_per_classifier = {l: torch.argmax(logits_per_classifier[l], dim=1).tolist()
                                 for l in logits_per_classifier}

    model_scores = get_score(labels, prediction_per_classifier)

    confidence_per_classifier = {l: torch.nn.Softmax(dim=1)(logits_per_classifier[l]).cpu().numpy()
                                 for l in logits_per_classifier.keys()}

    calibrated_confidence_per_classifier = temperature_calibration(confidence_per_classifier, labels)

    contrastive_predictions_per_classifier = get_contrastive_predictions(calibrated_confidence_per_classifier)
    contrastive_model_scores = get_score(labels, contrastive_predictions_per_classifier)

    print("----------------------------")
    print(dataset_key)
    print("----------------------------")
    print(json.dumps(model_scores, indent=4))
    print(json.dumps(contrastive_model_scores, indent=4))


def experiment_script(model_name_or_path: str, dataset_key: str):
    logits_per_classifier, labels = predict_script(model_name_or_path, dataset_key)

    prediction_per_classifier = {l: torch.argmax(logits_per_classifier[l], dim=1).tolist()
                                 for l in logits_per_classifier}

    model_scores = get_score(labels, prediction_per_classifier)
    print(json.dumps(model_scores, indent=4))

    confidence_per_classifier = {l: torch.nn.Softmax(dim=1)(logits_per_classifier[l]).cpu().numpy()
                                 for l in logits_per_classifier.keys()}

    # uncalibrated_scores = get_calibration_scores_per_layer(confidence_per_classifier, labels)
    # print(json.dumps(uncalibrated_scores, indent=4))

    calibrated_confidence_per_classifier = temperature_calibration(confidence_per_classifier, labels)

    # plot_overconfidence_histogram(calibrated_confidence_per_classifier, labels)
    get_data_for_layer_comparison_plot(calibrated_confidence_per_classifier, labels)
    a, b = get_layers_predictions_comparison(calibrated_confidence_per_classifier, labels, 0.2)
    plot_layers_comparison_histo(a, b)

    # calibrated_scores = get_calibration_scores_per_layer(calibrated_confidence_per_classifier, labels)
    # print(json.dumps(calibrated_scores, indent=4))

    # plot_reliability_diagram_per_layer(confidence_per_classifier, labels, "uncalibrated_")
    # plot_reliability_diagram_per_layer(calibrated_confidence_per_classifier, labels, "calibrated_")

    contrastive_predictions_per_classifier = get_contrastive_predictions(calibrated_confidence_per_classifier)
    contrastive_model_scores = get_score(labels, contrastive_predictions_per_classifier)
    print(json.dumps(contrastive_model_scores, indent=4))
    #
    # experiment_3(calibrated_confidence_per_classifier, labels)
    # get_contrastive_logits(logits_per_classifier)

    print("end")


def get_contrastive_score_diff_for_all_models():
    for model_dir in glob.glob("models/*"):
        dataset_key = os.path.basename(model_dir).split("_")[0]
        if dataset_key == "boolq":
            continue
        model_path = f'{model_dir}/data/'

        print_contrastive_results_comparison(model_path, dataset_key)


def test_swag():
    ending_names = ["ending0", "ending1", "ending2", "ending3"]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples["sent1"]]
        question_headers = examples["sent2"]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        return {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    swag = load_dataset("swag", "regular")

    tokenized_swag = swag.map(preprocess_function, batched=True)


def predict_script2(model_name_or_path: str, dataset_key: str, output_dir: str, split: str = "validation",
                    max_examples: Optional[int] = None) -> Tuple[Dict[int, torch.Tensor], List[int]]:
    tokenizer = get_tokenizer(model_name_or_path)
    dev_dataset = get_processed_dataset(dataset_key, split, tokenizer, max_examples)
    labels = dev_dataset['labels']
    sample_label = labels[0]
    if isinstance(sample_label, list):
        num_labels = len(sample_label)
    else:
        label_set = set(dev_dataset['labels'])
        num_labels = len(label_set)

    model = get_model(model_name_or_path, num_labels=num_labels)
    use_cpu = torch.cuda.is_available()
    model.to("cpu" if use_cpu else "cuda")

    data_collator: DataCollator = DataCollatorWithPadding(tokenizer)

    logits_per_classifier = {l: torch.empty(0, model.config.num_labels, device="cpu" if use_cpu else "cuda")
                             for l in model.config.classifiers_layers}
    with torch.no_grad():
        batch_size = 8
        for i in tqdm(range(0, len(dev_dataset), batch_size)):
            tokenized_inputs = data_collator(dev_dataset[i:min(i + batch_size, len(dev_dataset))]).to(
                "cpu" if use_cpu else "cuda")
            batch_logits = model(**tokenized_inputs).logits
            for layer, l_logits in batch_logits.items():
                logits_per_classifier[layer] = torch.concat([logits_per_classifier[layer], l_logits])

    _save_model_outputs_to_cache(model_name_or_path, dataset_key, output_dir,
                                 split, max_examples, logits_per_classifier, labels)

    return logits_per_classifier, labels


def combine_output():
    f1_path = "model_output/model_.._models_go-emotions_layer_12_not_shared_bert_data_go-emotions_layer_12_not_shared_bert_2_checkpoint-500__dataset_go-emotions_split_validation.json"
    f2_path = "model_output/model_.._models_go-emotions_layer_12_not_shared_bert_data_go-emotions_layer_12_not_shared_bert_2_checkpoint-9500__dataset_go-emotions_split_validation.json"

    logits_per_classifier = dict()
    labels = []
    with open(f1_path, 'r', encoding='utf-8') as f_500:
        data_500 = json.load(f_500)
        logits_per_classifier['500'] = data_500['logits_per_classifier']['12']
        labels = data_500['labels']

    with open(f2_path, 'r', encoding='utf-8') as f_9500:
        data_9500 = json.load(f_9500)
        logits_per_classifier['9500'] = data_9500['logits_per_classifier']['12']
        assert labels == data_9500['labels']

    output = {
        "logits_per_classifier": logits_per_classifier,
        "labels": labels
    }

    output_path = "model_output/go_emotions_500_9500.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f)



def load_file():
    file_path = "model_output/model_.._models_data_weighted_loss_mbert_mnli_checkpoint-3400__dataset_xnli_vi_split_validation.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logits_per_classifier, labels = data['logits_per_classifier'], data['labels']

    formatted_logits_per_classifier = dict()
    for k in logits_per_classifier.keys():
        logits = logits_per_classifier[k]
        logits = torch.tensor(logits)
        formatted_logits_per_classifier[int(k)] = logits

    logits_per_classifier = formatted_logits_per_classifier

    print(len(labels))
    confidence_per_classifier = {l: torch.nn.Softmax(dim=1)(logits_per_classifier[l]).cpu().numpy()
                                 for l in logits_per_classifier.keys()}
    calibrated_confidence_per_classifier = temperature_calibration(confidence_per_classifier, labels)
    plot_layer_comparison_heatmap(calibrated_confidence_per_classifier, labels, f'9500-500', ['12_8', '12_4', '12_2', '12_1'])

if __name__ == '__main__':
    # TODO:
    # 1) Use RoBERTa instead of BERT
    # 2) More Tasks
    # 3) Analyze
    # batch_run()
    # exit()
    # test_swag()
    # combine_output()
    # load_file()
    # exit()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    # region Train argparser
    parser_train = subparsers.add_parser('train', help='')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-m', '--model-name', type=str, default="bert-base-multilingual-cased")
    parser_train.add_argument('-d', '--dataset-key', required=True, type=str)
    parser_train.add_argument('-o', '--output-dir', required=True, type=str)
    parser_train.add_argument('-s', '--share-classifiers-weights', action='store_true')
    parser_train.add_argument('-c', '--max-train-examples', type=int, required=False, default=None)
    parser_train.add_argument('-l', '--classifiers-layers', nargs='+', required=False, default=[1, 2, 4, 8, 12],
                              type=int)

    # endregion

    # region Predict argparser
    parser_predict = subparsers.add_parser('predict', help='')
    parser_predict.set_defaults(which='predict')
    parser_predict.add_argument('-i', '--model-name-or-path', required=True, type=str)
    parser_predict.add_argument('-d', '--dataset-key', required=True, type=str)
    parser_predict.add_argument('-o', '--output-dir', required=True, type=str)

    # endregion

    # region Experiment argparser
    parser_experiment = subparsers.add_parser('experiment', help='')
    parser_experiment.set_defaults(which='experiment')
    parser_experiment.add_argument('-i', '--model-name-or-path', required=True, type=str)
    parser_experiment.add_argument('-d', '--dataset-key', required=True, type=str)
    # endregion

    set_seed(42)

    args = parser.parse_args()

    if args.which == "train":
        train_script(args.dataset_key, args.share_classifiers_weights, args.output_dir,
                     args.classifiers_layers, args.max_train_examples, args.model_name)
    elif args.which == "experiment":
        experiment_script(args.model_name_or_path, args.dataset_key)
    elif args.which == "predict":
        predict_script2(args.model_name_or_path, args.dataset_key, args.output_dir)

# TODO:
# Different checkpoints
# Zero-shot
# more weight to the upper layer

# Create load predictions function
# Edit the current pred with it, see fi ther are bugs
# create for multiple run indiffernet files
# create for zero shot in differnet file
# Parsing - UD
# Rub stuff so the loss for the upper layer is 90
# few shot vs english only

# Write finetune code, write ud code
# LM contrastive for Ud parsinf seq1seq

# TRu multilingual few shot!!! ust a few!! vand compare modeks!!