import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel, RobertaPreTrainedModel, RobertaModel

from ..utils.datasets import DependencyParsingModelOutput
from .biaffine import Biaffine


class BertForBiaffineParsing(BertPreTrainedModel):
    """
    Credit: G. Glavaš & I. Vulić
    Based on paper "Is Supervised Syntactic Parsing Beneficial for Language Understanding? An Empirical Investigation"
    (https://arxiv.org/pdf/2008.06788.pdf)
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.biaffine_arcs = Biaffine(n_in=config.hidden_size, bias_x=True, bias_y=False)
        self.biaffine_rels = Biaffine(n_in=config.hidden_size, n_out=config.num_labels, bias_x=True, bias_y=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_fn = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        word_starts=None,
        arc_labels=None,
        rel_labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # run through BERT encoder and get vector representations
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outs = self.dropout(outputs[0])

        word_outputs_deps = self._merge_subword_tokens(outs, word_starts)

        # adding the CLS representation as the representation for the "root" parse token
        word_outputs_heads = torch.cat([outputs[1].unsqueeze(1), word_outputs_deps], dim=1)

        arc_logits = self.biaffine_arcs(word_outputs_deps, word_outputs_heads)
        arc_logits = arc_logits.squeeze()

        rel_logits = self.biaffine_rels(word_outputs_deps, word_outputs_heads)
        rel_logits = rel_logits.permute(0, 2, 3, 1)

        loss = None
        if arc_labels is not None and rel_labels is not None:
            loss = self._get_loss(arc_logits, rel_logits, arc_labels, rel_labels, self.loss_fn)

        if len(arc_logits.shape) == 2:
            arc_logits = arc_logits.unsqueeze(0)

        if not return_dict:
            output = (
                arc_logits,
                rel_logits,
            ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DependencyParsingModelOutput(
            loss=loss,
            arc_logits=arc_logits,
            rel_logits=rel_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _merge_subword_tokens(self, subword_outputs, word_starts):
        instances = []
        max_seq_length = subword_outputs.shape[1]

        # handling instance by instance
        for i in range(len(subword_outputs)):
            subword_vecs = subword_outputs[i]
            word_vecs = []
            starts = word_starts[i]
            mask = starts.ne(self.config.pad_token_id)
            starts = starts[mask]
            for j in range(len(starts) - 1):
                if starts[j + 1] <= 0:
                    break

                start = starts[j]
                end = starts[j + 1]
                if start == end:
                    vecs_range = subword_vecs[start]
                    word_vecs.append(vecs_range.unsqueeze(0))
                else:
                    vecs_range = subword_vecs[start:end]
                    word_vecs.append(torch.mean(vecs_range, 0).unsqueeze(0))

            instances.append(word_vecs)

        t_insts = []
        zero_tens = torch.zeros(self.config.hidden_size).unsqueeze(0)
        zero_tens = zero_tens.to(self.device)

        for inst in instances:
            if len(inst) < max_seq_length:
                for i in range(max_seq_length - len(inst)):
                    inst.append(zero_tens)
            t_insts.append(torch.cat(inst, dim=0).unsqueeze(0))

        w_tens = torch.cat(t_insts, dim=0)
        return w_tens

    def _get_loss(self, arc_preds, rel_preds, labels_arc, labels_rel, loss_fn):
        if len(arc_preds.shape) == 2:
            arc_preds = arc_preds.unsqueeze(0)

        mask = labels_arc.ne(self.config.pad_token_id)
        arc_scores, arcs = arc_preds[mask], labels_arc[mask]
        loss = loss_fn(arc_scores, arcs)

        rel_scores, rels = rel_preds[mask], labels_rel[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        rel_loss = loss_fn(rel_scores, rels)
        loss += rel_loss

        return loss


class RobertaForBiaffineParsing(RobertaPreTrainedModel):
    """
    Credit: G. Glavaš & I. Vulić
    Based on paper "Is Supervised Syntactic Parsing Beneficial for Language Understanding? An Empirical Investigation"
    (https://arxiv.org/pdf/2008.06788.pdf)
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)

        self.biaffine_arcs = Biaffine(n_in=config.hidden_size, bias_x=True, bias_y=False)
        self.biaffine_rels = Biaffine(n_in=config.hidden_size, n_out=config.num_labels, bias_x=True, bias_y=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_fn = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        word_starts=None,
        arc_labels=None,
        rel_labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # run through RoBERTa encoder and get vector representations
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outs = self.dropout(outputs[0])

        word_outputs_deps = self._merge_subword_tokens(outs, word_starts)

        # adding the CLS representation as the representation for the "root" parse token
        word_outputs_heads = torch.cat([outputs[1].unsqueeze(1), word_outputs_deps], dim=1)

        arc_logits = self.biaffine_arcs(word_outputs_deps, word_outputs_heads)
        arc_logits = arc_logits.squeeze()

        rel_logits = self.biaffine_rels(word_outputs_deps, word_outputs_heads)
        rel_logits = rel_logits.permute(0, 2, 3, 1)

        loss = None
        if arc_labels is not None and rel_labels is not None:
            loss = self._get_loss(arc_logits, rel_logits, arc_labels, rel_labels, self.loss_fn)

        if len(arc_logits.shape) == 2:
            arc_logits = arc_logits.unsqueeze(0)

        if not return_dict:
            output = (
                arc_logits,
                rel_logits,
            ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DependencyParsingModelOutput(
            loss=loss,
            arc_logits=arc_logits,
            rel_logits=rel_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _merge_subword_tokens(self, subword_outputs, word_starts):
        instances = []
        max_seq_length = subword_outputs.shape[1]

        # handling instance by instance
        for i in range(len(subword_outputs)):
            subword_vecs = subword_outputs[i]
            word_vecs = []
            starts = word_starts[i]
            mask = starts.ne(self.config.pad_token_id)
            starts = starts[mask]
            for j in range(len(starts) - 1):
                if starts[j + 1] <= 0:
                    break

                start = starts[j]
                end = starts[j + 1]
                if start == end:
                    vecs_range = subword_vecs[start]
                    word_vecs.append(vecs_range.unsqueeze(0))
                else:
                    vecs_range = subword_vecs[start:end]
                    word_vecs.append(torch.mean(vecs_range, 0).unsqueeze(0))

            instances.append(word_vecs)

        t_insts = []
        zero_tens = torch.zeros(self.config.hidden_size).unsqueeze(0)
        zero_tens = zero_tens.to(self.device)

        for inst in instances:
            if len(inst) < max_seq_length:
                for i in range(max_seq_length - len(inst)):
                    inst.append(zero_tens)
            t_insts.append(torch.cat(inst, dim=0).unsqueeze(0))

        w_tens = torch.cat(t_insts, dim=0)
        return w_tens

    def _get_loss(self, arc_preds, rel_preds, labels_arc, labels_rel, loss_fn):
        if len(arc_preds.shape) == 2:
            arc_preds = arc_preds.unsqueeze(0)

        mask = labels_arc.ne(self.config.pad_token_id)
        arc_scores, arcs = arc_preds[mask], labels_arc[mask]
        loss = loss_fn(arc_scores, arcs)

        rel_scores, rels = rel_preds[mask], labels_rel[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        rel_loss = loss_fn(rel_scores, rels)
        loss += rel_loss

        return loss
