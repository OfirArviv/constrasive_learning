from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import add_start_docstrings, BertConfig
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BERT_START_DOCSTRING, BertPreTrainedModel, BertModel, \
    BERT_INPUTS_DOCSTRING, _CHECKPOINT_FOR_TOKEN_CLASSIFICATION, _CONFIG_FOR_DOC, _TOKEN_CLASS_EXPECTED_OUTPUT, \
    _TOKEN_CLASS_EXPECTED_LOSS, _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION, _SEQ_CLASS_EXPECTED_OUTPUT, \
    _SEQ_CLASS_EXPECTED_LOSS
from transformers.utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings, ModelOutput


@dataclass
class MultiHeadTokenClassifierOutput(TokenClassifierOutput):
    loss_per_classifier: Optional[Dict[int, torch.FloatTensor]] = None
    logits_per_classifier: Dict[int, torch.FloatTensor] = None


@dataclass
class MultiHeadSequenceClassifierOutput(SequenceClassifierOutput):
    loss_per_classifier: Optional[Dict[int, torch.FloatTensor]] = None
    logits_per_classifier: Dict[int, torch.FloatTensor] = None


class ContrastiveBertConfig(BertConfig):
    def __init__(
            self,
            num_labels: int = 2,
            classifiers_layers: Optional[List[int]] = None,
            share_classifiers_weights: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.classifiers_layers = classifiers_layers
        self.num_labels = num_labels
        self.share_classifiers_weights = share_classifiers_weights


@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
# class ContrastiveBertForTokenClassification(BertPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#
#     def __init__(self, config: ContrastiveBertConfig):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#
#         self.bert = BertModel(config, add_pooling_layer=False)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#
#         classifiers_layers = config.classifiers_layers
#
#         self.classifiers = nn.ModuleDict({k: nn.Linear(config.hidden_size, config.num_labels)
#                                           for k in classifiers_layers})
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
#         expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
#     )
#     def forward(
#             self,
#             input_ids: Optional[torch.Tensor] = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             token_type_ids: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.Tensor] = None,
#             head_mask: Optional[torch.Tensor] = None,
#             inputs_embeds: Optional[torch.Tensor] = None,
#             labels: Optional[torch.Tensor] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         output_hidden_states = True
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         # sequence_output = outputs[0]
#         # sequence_output = self.dropout(sequence_output)
#         sequence_outputs = {k: self.dropout(outputs.hidden_states[k]) for k in self.classifiers.keys()}
#
#         # logits = self.classifier(sequence_output)
#         logits_per_classifier = {k: self.classifiers[k](sequence_outputs[k]) for k in self.classifiers.keys()}
#
#         loss = None
#         loss_per_classifier = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             loss_per_classifier = {k: loss_fct(logits_per_classifier[k].view(-1, self.num_labels), labels.view(-1))
#                                    for k in self.classifiers.keys()}
#             loss = torch.stack(list(loss_per_classifier.values())).sum()
#
#         if not return_dict:
#             # output = (logits,) + outputs[2:]
#             # return ((loss,) + output) if loss is not None else output
#             return loss if loss is not None else outputs
#
#         return MultiHeadTokenClassifierOutput(
#             loss=loss,
#             loss_per_classifier=loss_per_classifier,
#             logits_per_classifier=logits_per_classifier,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#

class ContrastiveBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.activation = nn.Tanh()

        self.classifiers_layers = config.classifiers_layers

        self.share_classifiers_weights = config.share_classifiers_weights
        if self.share_classifiers_weights:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifiers = nn.ModuleDict({str(k): nn.Linear(config.hidden_size, config.num_labels)
                                              for k in self.classifiers_layers})

        # Initialize weights and apply final processing
        self.post_init()
        self.i = 0

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        pooled_outputs = {k: self.dropout(self.activation(outputs.hidden_states[k][:, 0])) for k in
                          self.classifiers_layers}

        if self.share_classifiers_weights:
            logits_per_classifier = {k: self.classifier(pooled_outputs[k]) for k in self.classifiers_layers}
        else:
            logits_per_classifier = {k: self.classifiers[str(k)](pooled_outputs[k]) for k in self.classifiers_layers}

        loss = None
        loss_per_classifier = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # loss = loss_fct(logits.squeeze(), labels.squeeze())
                    loss_per_classifier = {
                        k: loss_fct(logits_per_classifier[k].squeeze(), labels.squeeze())
                        for k in self.classifiers_layers}
                else:
                    # loss = loss_fct(logits, labels)
                    loss_per_classifier = {
                        k: loss_fct(logits_per_classifier[k], labels)
                        for k in self.classifiers_layers}
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_per_classifier = {
                    k: loss_fct(logits_per_classifier[k].view(-1, self.num_labels), labels.view(-1))
                    for k in self.classifiers_layers}
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                # loss = loss_fct(logits, labels)
                loss_per_classifier = {
                    k: loss_fct(logits_per_classifier[k], labels.float())
                    for k in self.classifiers_layers}
            else:
                raise NotImplementedError(self.config.problem_type)

            loss = torch.stack(list(loss_per_classifier.values())).sum()

        if not return_dict:
            # output = (logits,) + outputs[2:]
            output = (logits_per_classifier,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultiHeadSequenceClassifierOutput(
            loss=loss,
            loss_per_classifier=loss_per_classifier,
            logits=logits_per_classifier,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
