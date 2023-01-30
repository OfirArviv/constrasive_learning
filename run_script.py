import argparse
import itertools
import json
from statistics import median
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from netcal.presentation import ReliabilityDiagram
from netcal.scaling import TemperatureScaling
import evaluate
import numpy as np
from datasets import load_dataset, Dataset
from netcal.metrics.confidence import ECE
from transformers import AutoTokenizer, DataCollatorWithPadding, \
    PreTrainedTokenizerBase, PreTrainedModel, set_seed
import torch

from bert_modeling import ContrastiveBertConfig, \
    ContrastiveBertForSequenceClassification
from trainer import train


# region Dataset processing
def basic_sequence_classification_preprocess_dataset(examples,
                                                     tokenizer: PreTrainedTokenizerBase,
                                                     input_column: str,
                                                     label_column: str,
                                                     max_source_length: int,
                                                     max_target_length: int):
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


def preprocess_sst_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    source_column = "sentence"
    target_column = "label"
    max_seq_length = 512
    return dataset.map(lambda examples: basic_sequence_classification_preprocess_dataset(examples,
                                                                                         tokenizer,
                                                                                         source_column,
                                                                                         target_column,
                                                                                         max_seq_length,
                                                                                         max_seq_length),
                       remove_columns=dataset.column_names,
                       batched=True,
                       load_from_cache_file=False)


def snli_sequence_classification_preprocess_dataset(examples,
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


def preprocess_snli_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    input_column_1 = "premise"
    input_column_2 = "hypothesis"
    target_column = "label"
    max_seq_length = 512
    return dataset.map(lambda examples: snli_sequence_classification_preprocess_dataset(examples,
                                                                                        tokenizer,
                                                                                        input_column_1,
                                                                                        input_column_2,
                                                                                        target_column,
                                                                                        max_seq_length),
                       remove_columns=dataset.column_names,
                       batched=True,
                       load_from_cache_file=False)


def boolq_sequence_classification_preprocess_dataset(examples,
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


def preprocess_boolq_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    question_column = "question"
    context_column = "passage"
    target_column = "label"
    max_seq_length = 512
    return dataset.map(lambda examples: boolq_sequence_classification_preprocess_dataset(examples,
                                                                                         tokenizer,
                                                                                         question_column,
                                                                                         context_column,
                                                                                         target_column,
                                                                                         max_seq_length),
                       remove_columns=dataset.column_names,
                       batched=True,
                       load_from_cache_file=False)


def get_processed_dataset(dataset_key: str, split: str, tokenizer: PreTrainedTokenizerBase,
                          size: Optional[int] = None) -> Dataset:
    if dataset_key == "sst2":
        dataset_preprocess_func = preprocess_sst_dataset
        dataset = load_dataset(dataset_key, split=split)
    elif dataset_key == "snli":
        dataset_preprocess_func = preprocess_snli_dataset
        dataset = load_dataset(dataset_key, split=split)
    elif dataset_key == "boolq":
        dataset_preprocess_func = preprocess_boolq_dataset
        dataset = load_dataset("super_glue", name="boolq", split=split)
    else:
        raise NotImplementedError(dataset_key)

    if size is not None:
        size = min(size, len(dataset))
        dataset = dataset.shuffle(seed=42).select(range(size))
    dataset = dataset_preprocess_func(dataset, tokenizer)

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
                                                   average=fscore_average)
                         for l in prediction_per_classifier}

    acc_score = evaluate.load("accuracy")
    acc_per_classifier = {f'{l}_acc': acc_score.compute(references=labels, predictions=prediction_per_classifier[l])
                          for l in prediction_per_classifier}

    res_dict = f1_per_classifier
    res_dict.update(acc_per_classifier)

    return res_dict


def get_model_and_tokenizer(model_name_or_path: str,
                            num_labels: Optional[int] = None,
                            share_classifiers_weights: Optional[bool] = None
                            ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    config = ContrastiveBertConfig.from_pretrained(model_name_or_path)
    if config.classifiers_layers is None:
        assert share_classifiers_weights is not None, 'share_classifiers_weights is None. Config does not contain ' \
                                                      'param share_classifiers_weights and thus it needs to be provided'
        assert num_labels is not None
        config.num_labels = num_labels
        config.classifiers_layers = [4, 8, 12]
        config.share_classifiers_weights = share_classifiers_weights

    config.num_labels = 3

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = ContrastiveBertForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    return model, tokenizer


def train_script(dataset_key: str, share_classifiers_weights: bool, output_dir: str):
    num_labels = 3 if dataset_key == "snli" else 2

    model, tokenizer = get_model_and_tokenizer("bert-base-cased", num_labels, share_classifiers_weights)
    data_collator = DataCollatorWithPadding(tokenizer)

    use_cpu = False
    model.to("cpu" if use_cpu else "cuda")

    train_dataset = get_processed_dataset(dataset_key, "train", tokenizer, 10000)
    dev_dataset = get_processed_dataset(dataset_key, "validation", tokenizer, 2000)

    train(model, tokenizer, train_dataset, dev_dataset, data_collator, output_dir, 5, no_cuda=use_cpu)


def predict_script(model_name_or_path: str, dataset_key: str) -> Tuple[Dict[int, torch.Tensor], List[int]]:
    model, tokenizer = get_model_and_tokenizer(model_name_or_path)
    model.cuda()

    dev_dataset = get_processed_dataset(dataset_key, "validation", tokenizer, 2000)
    data_collator = DataCollatorWithPadding(tokenizer)

    logits_per_classifier = {l: torch.empty(0, model.config.num_labels, device="cuda")
                             for l in model.config.classifiers_layers}
    with torch.no_grad():
        batch_size = 8
        for i in range(0, len(dev_dataset), batch_size):
            tokenized_inputs = data_collator(dev_dataset[i:min(i + batch_size, len(dev_dataset))]).to('cuda')
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

def get_lower_layer_overconfidence(higher_layer_confidences: np.ndarray, lower_layer_confidences: np.ndarray):
    lower_layer_prob_argmax = np.argmax(lower_layer_confidences)
    lower_layer_prob = lower_layer_confidences[lower_layer_prob_argmax]
    higher_layer_prob = higher_layer_confidences[lower_layer_prob_argmax]

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


if __name__ == '__main__':
    # TODO:
    # 1) Use RoBERTa instead of BERT
    # 2) More Tasks
    # 3) Analyze

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

    args = parser.parse_args()

    set_seed(42)

    if args.which == "train":
        train_script(args.dataset_key, args.share_classifiers_weights, args.output_dir)
    if args.which == "experiment":
        experiment_script(args.model_name_or_path, args.dataset_key)
