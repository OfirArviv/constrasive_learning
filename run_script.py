import argparse
import glob
import itertools
import json
import os
from statistics import median
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
    PreTrainedTokenizerBase, PreTrainedModel, set_seed
import torch
import math

from bert_modeling import ContrastiveBertConfig, \
    ContrastiveBertForSequenceClassification
from trainer import train
import pandas as pd

import os

if os.path.exists('/dccstor'):
    os.environ['TRANSFORMERS_CACHE'] = '/dccstor/sum-datasets/users/ofir.arviv/transformers_cache'
    os.environ['HF_HOME'] = '/dccstor/sum-datasets/ofir.arviv/transformers_cache'


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

    else:
        raise NotImplementedError(dataset_key)

    if size is not None:
        size = min(size, len(dataset))
        dataset = dataset.shuffle(seed=42).select(range(size))
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
    acc_per_classifier = {f'{l}_acc': acc_score.compute(references=labels, predictions=prediction_per_classifier[l])
                          for l in prediction_per_classifier}

    res_dict = f1_per_classifier
    # res_dict.update(acc_per_classifier)

    return res_dict


def get_model(model_name_or_path: str,
              num_labels: Optional[int] = None,
              share_classifiers_weights: Optional[bool] = None
              ) -> PreTrainedModel:
    config = ContrastiveBertConfig.from_pretrained(model_name_or_path)
    if config.classifiers_layers is None:
        assert share_classifiers_weights is not None, 'share_classifiers_weights is None. Config does not contain ' \
                                                      'param share_classifiers_weights and thus it needs to be provided'
        assert num_labels is not None
        config.num_labels = num_labels
        config.classifiers_layers = [4, 8, 12]
        config.share_classifiers_weights = share_classifiers_weights

    # TODO: For some reason this config isnt saved
    config.num_labels = num_labels

    model = ContrastiveBertForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    return model


def get_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def train_script(dataset_key: str, share_classifiers_weights: bool, output_dir: str):
    model_name = "bert-base-cased"
    tokenizer = get_tokenizer(model_name)

    train_dataset = get_processed_dataset(dataset_key, "train", tokenizer, 10000)
    dev_dataset = get_processed_dataset(dataset_key, "validation", tokenizer, 2000)

    num_labels = len(set(train_dataset['labels']))

    model = get_model(model_name, num_labels, share_classifiers_weights)
    data_collator = DataCollatorWithPadding(tokenizer)

    use_cpu = False
    model.to("cpu" if use_cpu else "cuda")

    train(model, tokenizer, train_dataset, dev_dataset, data_collator, output_dir, 5, no_cuda=use_cpu)


def predict_script(model_name_or_path: str, dataset_key: str, split: str = "validation",
                   max_examples: int = 2000) -> Tuple[Dict[int, torch.Tensor], List[int]]:
    use_cpu = True
    tokenizer = get_tokenizer(model_name_or_path)

    dev_dataset = get_processed_dataset(dataset_key, split, tokenizer, max_examples)
    num_labels = len(set(dev_dataset['labels']))

    model = get_model(model_name_or_path, num_labels)
    model.to("cpu" if use_cpu else "cuda")

    data_collator = DataCollatorWithPadding(tokenizer)

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

    return logits_per_classifier, dev_dataset['labels']


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

# TODO: We check the case where both layers agree.
#  There are other cases, for example, where the lower layer thinks differently from the higher layer,
#  but we will check it only in the future.
def get_lower_layer_overconfidence(higher_layer_confidences: np.ndarray, lower_layer_confidences: np.ndarray):
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


# endregion


def exp():
    # TODO: function to caclualte t he conidence score, and do normalization

    # First analyze: Divide each layer couple (x,y) to "both correct", "layer x correct", "layer y correct","both wrong"
    data_first_analyze = dict()
    for layer_1, layer_2 in itertools.combinations(prediction_per_classifier.keys(), 2):
        layer_1_correct_count = 0
        layer_2_correct_count = 0
        both_layers_correct_count = 0
        none_layers_correct_count = 0

        for i, label in enumerate(labels):
            layer_1_pred = prediction_per_classifier[layer_1][i]
            layer_2_pred = prediction_per_classifier[layer_2][i]

            if label == layer_1_pred and label == layer_2_pred:
                both_layers_correct_count += 1
            elif label == layer_1_pred:
                layer_1_correct_count += 1
            elif label == layer_2_pred:
                layer_2_correct_count += 1
            else:
                none_layers_correct_count += 1

        data_first_analyze[f'{layer_1}_{layer_2}'] = {
            f'{layer_1} correct': layer_1_correct_count,
            f'{layer_2} correct': layer_2_correct_count,
            f'both correct': both_layers_correct_count,
            f'none correct': none_layers_correct_count
        }

    # Second analyze: When both are wrong, what are the confidence levels?
    # What I'm looking for?
    # When They are both wrong
    # If the label 0, I want confidence[0] of layer 1 to be close to 50 and layer 2 to be close to 0
    # If the label 1, I want confidence[1] of layer 1 to be close to 50 and layer 2 to be close to 0
    data_second_analyze = dict()
    for layer_1, layer_2 in itertools.combinations(prediction_per_classifier.keys(), 2):
        data_second_analyze[f'{layer_1}_{layer_2}'] = []
        for i, label in enumerate(labels):
            layer_1_pred = prediction_per_classifier[layer_1][i]
            layer_2_pred = prediction_per_classifier[layer_2][i]

            if label != layer_1_pred and label != layer_2_pred:
                layer_1_confidence = confidence_per_classifier[layer_1][i][label]
                layer_2_confidence = confidence_per_classifier[layer_2][i][label]

                data_second_analyze[f'{layer_1}_{layer_2}'].append((layer_1_confidence, layer_2_confidence))

    for k, v in data_second_analyze.items():
        cnt = 0
        diff_list = []
        for (p_l1, p_l2) in v:
            if p_l1 > p_l2:
                cnt += 1
                diff_list.append(p_l1 - p_l2)

        print(f'{k}: cnt: {cnt}/{len(v)}, diff median: {median(diff_list)}')


# If we look at the cases the higher layer is not certain, what is the confidence of the lower level?
def experiment_3(confidences_per_classifier: Dict[int, np.ndarray], labels: List[int]):
    data = dict()
    data_list = dict()
    for layer_1, layer_2 in itertools.combinations(confidence_per_classifier.keys(), 2):
        higher_layer_correct_count = 0
        lower_layer_correct_count = 0
        both_layers_correct_count = 0
        none_layers_correct_count = 0

        higher_layer = layer_1 if layer_1 > layer_2 else layer_2
        lower_layer = layer_1 if higher_layer == layer_2 else layer_2

        higher_layer_confidence = confidence_per_classifier[higher_layer]
        lower_layer_confidence = confidence_per_classifier[lower_layer]

        for i, (label_0_conf, label_1_conf) in enumerate(higher_layer_confidence):
            # The case the classifier is not certain about the prediction
            if 0.3 < label_0_conf < 0.7:
                lower_layer_label_0_conf, lower_layer_layer_1_conf = lower_layer_confidence[i]
                # The case the lower layer is confidence about the prediction
                if lower_layer_label_0_conf < 1.1 or lower_layer_label_0_conf > 0:
                    higher_layer_prediction = np.argmax(higher_layer_confidence[i])
                    lower_layer_prediction = np.argmax(lower_layer_confidence[i])
                    label = labels[i]

                    if label == higher_layer_prediction and label == lower_layer_prediction:
                        both_layers_correct_count += 1
                    elif label == higher_layer_prediction:
                        higher_layer_correct_count += 1
                    elif label == lower_layer_prediction:
                        lower_layer_correct_count += 1
                    else:
                        none_layers_correct_count += 1

        data[f'{higher_layer}_{lower_layer}'] = {
            f'{higher_layer} correct': higher_layer_correct_count,
            f'{lower_layer} correct': lower_layer_correct_count,
            f'both correct': both_layers_correct_count,
            f'none correct': none_layers_correct_count
        }

    print("a")


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
            f'none correct': list()
        }

        data_higher_layer_confidence[f'{higher_layer}_{lower_layer}'] = {
            f'{higher_layer} correct': list(),
            f'{lower_layer} correct': list(),
            f'both correct': list(),
            f'none correct': list()
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
            if lower_layer_overconfidence > overconfidence_threshold and higher_layer_prediction_confidence > higher_layer_confidence_threshold:
                label = labels[i]

                if label == higher_layer_prediction and label == lower_layer_prediction:
                    curr_overconfidence_data['both correct'].append(lower_layer_overconfidence)
                    curr_higher_layer_confidence_data['both correct'].append(higher_layer_prediction_confidence)
                elif label == higher_layer_prediction:
                    curr_overconfidence_data[f'{higher_layer} correct'].append(lower_layer_overconfidence)
                    curr_higher_layer_confidence_data[f'{higher_layer} correct'].append(
                        higher_layer_prediction_confidence)
                elif label == lower_layer_prediction:
                    curr_overconfidence_data[f'{lower_layer} correct'].append(lower_layer_overconfidence)
                    curr_higher_layer_confidence_data[f'{lower_layer} correct'].append(
                        higher_layer_prediction_confidence)
                else:
                    curr_overconfidence_data[f'none correct'].append(lower_layer_overconfidence)
                    curr_higher_layer_confidence_data[f'none correct'].append(higher_layer_prediction_confidence)

    return data_overconfidence, data_higher_layer_confidence


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def plot_layer_comparison_heatmap(confidences_per_classifier: Dict[int, np.ndarray],
                                  labels: List[int],
                                  dataset_name: str):
    y_bins = list(np.round(np.linspace(0, 0.9, 10), 2))
    x_bins = list(np.round(np.linspace(0, 0.9, 10), 2))

    # X-axis - overconfidence
    # Y-axis - upper layer confidence
    # Z-axis - (None correct - both correct)
    df = pd.DataFrame(columns=y_bins, index=x_bins)
    df = df.rename_axis("overconfidence", axis=0)
    df = df.rename_axis("higher layer confidence", axis=1)
    df = df.fillna(0)

    df_both_correct_cnt = df.copy(deep=True)
    df_both_incorrect_cnt = df.copy(deep=True)

    annotated_df = df.copy(deep=True)

    data_overconfidence, data_higher_layer_confidence = \
        get_layers_predictions_comparison(confidences_per_classifier, labels, 0, 0)

    fig, axs = plt.subplots(ncols=3, figsize=(21, 7))
    for i, layers_selection in enumerate(["12_8", "12_4", "8_4"]):
        for prediction_axis in ["both correct", "none correct"]:
            selected_overconfidence_data = data_overconfidence[layers_selection][prediction_axis]
            selected_higher_layer_confidence_data = data_higher_layer_confidence[layers_selection][prediction_axis]
            for higher_layer_confidence, lower_layer_overconfidence in zip(selected_higher_layer_confidence_data,
                                                                           selected_overconfidence_data):
                higher_layer_confidence_bin = int(math.floor(higher_layer_confidence * 100) / 10) * 10 / 100.0
                lower_layer_overconfidence_bin = int(math.floor(lower_layer_overconfidence * 100) / 10) * 10 / 100.0
                lower_layer_overconfidence_bin = min(lower_layer_overconfidence_bin, 0.9)
                if prediction_axis == "both correct":
                    df.loc[lower_layer_overconfidence_bin, higher_layer_confidence_bin] -= 1
                    df_both_correct_cnt.loc[lower_layer_overconfidence_bin, higher_layer_confidence_bin] += 1
                else:
                    df.loc[lower_layer_overconfidence_bin, higher_layer_confidence_bin] += 1
                    df_both_incorrect_cnt.loc[lower_layer_overconfidence_bin, higher_layer_confidence_bin] += 1

        for col, row in itertools.product(df.columns, df.index):
            both_correct_cnt = df_both_correct_cnt.loc[row, col]
            both_incorrect_cnt = df_both_incorrect_cnt.loc[row, col]
            diff_cnt = both_incorrect_cnt-both_correct_cnt

            diff_cnt_formatted = human_format(diff_cnt)
            both_correct_cnt_formatted = human_format(both_correct_cnt)
            both_incorrect_cnt_formatted = human_format(both_incorrect_cnt)
            annt = f'{diff_cnt_formatted} \n ({both_incorrect_cnt_formatted},\n{both_correct_cnt_formatted})'
            annotated_df.loc[row, col] = annt

        df = df.clip(upper=99, lower=-99)
        ax = sns.heatmap(df, annot=annotated_df, linewidth=0.5, center=0, xticklabels=True, yticklabels=True,
                         vmax=3, vmin=-3, ax=axs[i], cbar=False, fmt='')
        ax.set(title=f'{dataset_name} - Layers {layers_selection}: None Correct - Both Correct')
    plt.tight_layout()
    plt.show()


def plot_layer_comparison_heatmap_for_all_models():
    for model_dir in glob.glob("models/*"):
        dataset_key = os.path.basename(model_dir).split("_")[0]
        if dataset_key in ["boolq"]:
            continue
        model_path = f'{model_dir}/data/'
        print(dataset_key)
        logits_per_classifier, labels = predict_script(model_path, dataset_key, max_examples=1000000, split="validation")
        print(len(labels))
        confidence_per_classifier = {l: torch.nn.Softmax(dim=1)(logits_per_classifier[l]).cpu().numpy()
                                     for l in logits_per_classifier.keys()}
        calibrated_confidence_per_classifier = temperature_calibration(confidence_per_classifier, labels)
        plot_layer_comparison_heatmap(calibrated_confidence_per_classifier, labels, dataset_key)




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


if __name__ == '__main__':
    # TODO:
    # 1) Use RoBERTa instead of BERT
    # 2) More Tasks
    # 3) Analyze
    # plot_layer_comparison_heatmap_for_all_models()
    # exit()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    # region Train argparser
    parser_train = subparsers.add_parser('train', help='')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-d', '--dataset-key', required=True, type=str)
    parser_train.add_argument('-o', '--output-dir', required=True, type=str)
    parser_train.add_argument('-s', '--share-classifiers-weights', action='store_true')

    # endregion

    # region Expriment argparser
    parser_experiment = subparsers.add_parser('experiment', help='')
    parser_experiment.set_defaults(which='experiment')
    parser_experiment.add_argument('-i', '--model-name-or-path', required=True, type=str)
    parser_experiment.add_argument('-d', '--dataset-key', required=True, type=str)
    # endregion

    set_seed(42)

    args = parser.parse_args()

    if args.which == "train":
        train_script(args.dataset_key, args.share_classifiers_weights, args.output_dir)
    if args.which == "experiment":
        experiment_script(args.model_name_or_path, args.dataset_key)
