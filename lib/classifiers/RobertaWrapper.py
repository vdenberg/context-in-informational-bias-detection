from transformers import BertPreTrainedModel, BertModel
from torch.nn import Dropout, Linear
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch import nn
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CyclicLR
import os, pickle
import numpy as np
from lib.utils import get_torch_device, to_tensor, to_batches
from lib.evaluate.Eval import my_eval
from torch.nn import CrossEntropyLoss, MSELoss, Embedding, Dropout, Linear, Sigmoid, LSTM
from transformers.configuration_roberta import RobertaConfig

from transformers.modeling_roberta import RobertaModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from allennlp.modules import TimeDistributed
from sklearn.metrics.pairwise import cosine_similarity

# helpers
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, my_id, input_ids, input_mask, segment_ids, label_id):
        self.my_id = my_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def load_features(fp, batch_size, sampler='sequential'):
    with open(fp, "rb") as f:
        ids, data, labels = to_tensor(pickle.load(f))
    batches = to_batches(data, batch_size=batch_size, sampler=sampler)
    return ids, batches, labels


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHeadwTDFF(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHeadwTDFF, self).__init__()
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = TimeDistributed(Linear(config.hidden_size, config.hidden_size))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        #self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.out_proj = TimeDistributed(Linear(config.hidden_size, config.num_labels))

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS]) #todo: process this!!!!
        x = x.unsqueeze(dim=0)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = x.squeeze()
        return x

#@add_start_docstrings("""RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
 #   on top of the pooled output) e.g. for GLUE tasks. """,
 #                     ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.sigm = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                labels=None, ssc=False):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        logits = self.classifier(sequence_output)
        #probs = self.sigm(logits)

        #outputs = (logits, probs, sequence_output) + outputs[2:]
        outputs = (logits, pooled_output, sequence_output) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


#@add_start_docstrings("""RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
 #   on top of the pooled output) e.g. for GLUE tasks. """,
 #                     ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForSequenceClassificationWithTDFF(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassificationWithTDFF, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHeadwTDFF(config)

        self.sigm = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                labels=None, ssc=False):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        logits = self.classifier(sequence_output)

        #probs = self.sigm(logits)

        #outputs = (logits, probs, sequence_output) + outputs[2:]
        outputs = (logits, pooled_output, sequence_output) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


#@add_start_docstrings("""Roberta Model with a token classification head on top (a linear layer on top of
#    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
#                      ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        self.sigm = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        """
            Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
                **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
                    Sequence of hidden-states at the output of the last layer of the model.
                **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
                    Last layer hidden-state of the first token of the sequence (classification token)
                    further processed by a Linear layer and a Tanh activation function. The Linear
                    layer weights are trained from the next sentence prediction (classification)
                    objective during Bert pretraining. This output is usually *not* a good summary
                    of the semantic content of the input, you're often better with averaging or pooling
                    the sequence of hidden-states for the whole input sequence.
                **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
                    list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
                    of shape ``(batch_size, sequence_length, hidden_size)``:
                    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
                **attentions**: (`optional`, returned when ``config.output_attentions=True``)
                    list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
                    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        """
        sequence_output = outputs[0]  # sequence of hidden-states at the output of the last layer of the model
        pooled_output = outputs[1]   # last layer hidden-state of the first token of the sequence

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        probs = self.sigm(logits)

        outputs = (logits, probs, pooled_output, sequence_output) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class Inferencer():
    def __init__(self, reports_dir, logger, device, use_cuda):
        self.device = device
        self.reports_dir = reports_dir
        self.logger = logger
        self.device = device
        self.use_cuda = use_cuda

    def predict(self, model, data, return_embeddings=False, emb_type='poolbert', output_mode='sent_clf'):
        model.to(self.device)
        model.eval()

        preds = []
        embeddings = []
        labels = []

        for step, batch in enumerate(data):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, label_ids = batch
            # labels.extend(label_ids)

            with torch.no_grad():
                # print(input_mask)
                outputs = model(input_ids, input_mask, labels=None)
                logits = outputs[0]

            # of last hidden state with size (batch_size, sequence_length, hidden_size)
            # where batch_size=1, sequence_length=95, hidden_size=768)
            # take average of sequence, size (batch_size, hidden_size)

            if return_embeddings:
                pooled_output, sequence_output, hidden_states = outputs[2], outputs[3], outputs[-1]
                '''
                **sequence_output**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
                    Sequence of hidden-states at the output of the last layer of the model.
                **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
                    Last layer hidden-state of the first token of the sequence (classification token)
                    further processed by a Linear layer and a Tanh activation function. The Linear
                    layer weights are trained from the next sentence prediction (classification)
                    objective during Bert pretraining. This output is usually *not* a good summary
                    of the semantic content of the input, you're often better with averaging or pooling
                    the sequence of hidden-states for the whole input sequence.
                **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
                    list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
                    of shape ``(batch_size, sequence_length, hidden_size)``:
                    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
                '''
                if emb_type == 'poolbert':
                    emb_output = pooled_output
                elif emb_type == "avbert":
                    emb_output = sequence_output.mean(axis=1)
                elif emb_type == "unpoolbert":
                    emb_output = sequence_output[:, 0, :]
                elif emb_type == "crossbert":
                    hidden_states = torch.stack(hidden_states[:-1])
                    emb_output = hidden_states[:, :, 0, :].mean(dim=0)
                elif emb_type == "cross4bert":
                    hidden_states = hidden_states[:-1]
                    hidden_states = torch.stack(hidden_states[-4:])
                    emb_output = hidden_states[:, :, 0, :].mean(dim=0)

                if self.use_cuda:
                    emb_output = list(emb_output[0].detach().cpu().numpy())  # .detach().cpu() necessary here on gpu

                else:
                    self.logger.info("NOT USING CUDA")
                    emb_output = list(emb_output[0].numpy())

                #print(emb_output)
                embeddings.append(emb_output)

            logits = logits.detach().cpu().numpy()
            #probs = probs.detach().cpu().numpy()

            if output_mode == 'tok_clf':
                pred = [list(p) for p in np.argmax(logits, axis=2)]

            elif output_mode == 'sent_clf':

                if len(logits.shape) == 1:
                    logits = logits.unsqueeze()
                try:
                    pred = np.argmax(logits, axis=1)
                except:
                    print(logits)
                    print(logits.shape)
                    exit(0)

            preds.extend(pred)

        # rep_sim = sum(rep_sim) / len(rep_sim)

        model.train()
        if return_embeddings:
            return embeddings
        else:
            return preds, labels

    def evaluate(self, model, data, labels, av_loss=None, set_type='dev', name='Basil', output_mode='sent_clf'):
        preds, rep_sim = self.predict(model, data, output_mode=output_mode)
        # print('Evaluation these predictions:', len(preds), len(preds[0]), preds[:2])
        # print('Evaluation above predictions with these labels:', len(labels), len(labels[0]), labels[:2])
        if output_mode == 'tok_clf':
            labels = labels.numpy().flatten()
            preds = np.asarray(preds)
            preds = np.reshape(preds, labels.shape)
        else:
            labels = labels.numpy

        print(preds)
        print(labels)
        if len(preds) != len(labels):
            print(f'Sizes of {set_type} not equal')
            print(preds, labels)
            print(len(preds)) #, len(preds[0]))
            print(len(labels)) #, len(labels[0]))
            exit(0)

        metrics_dict, metrics_string = my_eval(labels, preds, set_type=set_type, av_loss=av_loss, name=name, rep_sim=rep_sim, opmode=output_mode)

        # output_eval_file = os.path.join(self.reports_dir, f"{name}_eval_results.txt")
        # self.logger.info(f'{metrics_string}')
        # with open(output_eval_file, 'w') as f:
        #    f.write{metrics_string + '\n'}

        return metrics_dict, metrics_string


def save_model(model_to_save, model_dir, identifier):
    output_dir = os.path.join(model_dir, identifier)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(output_dir, "config.json")

    model_to_save = model_to_save.module if hasattr(model_to_save,
                                                    'module') else model_to_save  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    # test again