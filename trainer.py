from typing import Optional
import numpy as np
import evaluate
from datasets import Dataset
from transformers import TrainingArguments, Trainer, PreTrainedModel, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, EarlyStoppingCallback


def sequence_classification_preprocess_dataset(examples,
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


def fscore_eval_func(eval_preds: EvalPrediction):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    fscore = evaluate.load("f1")

    res = dict()
    for layer in preds.keys():
        logits = preds[layer]
        layer_preds = np.argmax(logits, axis=1)
        layer_res = fscore.compute(references=labels, predictions=layer_preds)
        for metric, val in layer_res.items():
            res[f'layer_{layer}_{metric}'] = val

    return res


def train(model: PreTrainedModel,
          tokenizer: PreTrainedTokenizerBase,
          train_dataset: Dataset,
          eval_dataset: Optional[Dataset],
          data_collator: DataCollator,
          output_dir: str,
          num_epochs: int,
          optimizer_name: str = "adamw_hf",
          learning_rate: float = 3e-5,
          lr_scheduler_type: str = "linear",
          debug=False):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        optim=optimizer_name,
        lr_scheduler_type=lr_scheduler_type,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="layer_12_f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset.select(range(100)) if debug else eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=fscore_eval_func,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
