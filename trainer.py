from typing import Optional, Callable
import numpy as np
import evaluate
from datasets import Dataset
from transformers import TrainingArguments, Trainer, PreTrainedModel, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, EarlyStoppingCallback


def get_fscore_eval_func(num_labels: int) -> Callable:
    if num_labels == 2:
        fscore_average = "binary"
    else:
        fscore_average = "weighted"

    def fscore_eval_func(eval_preds: EvalPrediction):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        fscore = evaluate.load("f1")

        res = dict()
        for layer in preds.keys():
            logits = preds[layer]
            layer_preds = np.argmax(logits, axis=1)
            layer_res = fscore.compute(references=labels, predictions=layer_preds, average=fscore_average)
            for metric, val in layer_res.items():
                res[f'layer_{layer}_{metric}'] = val

        return res

    return fscore_eval_func


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
          no_cuda: bool = False,
          debug=False):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        optim=optimizer_name,
        lr_scheduler_type=lr_scheduler_type,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="layer_12_f1",
        no_cuda=no_cuda,
        # fp16=True,
        # gradient_accumulation_steps=4,
        # eval_accumulation_steps=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset.select(range(100)) if debug else eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_fscore_eval_func(model.config.num_labels),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
