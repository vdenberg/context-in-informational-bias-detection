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
from torch.nn import CrossEntropyLoss, MSELoss, Embedding, Dropout, Linear, Sigmoid, LSTM


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


class BertForTokenClassification(BertPreTrainedModel):
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

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sigm = nn.Sigmoid()

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        probs = self.sigm(logits)

        outputs = (logits, probs, sequence_output, outputs[1]) + outputs[2:]  # add hidden states and attention if they are here
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

        return outputs  # (loss), logits, probs, scores, (hidden_states), (attentions)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, self.config.num_labels)
        self.sigm = nn.Sigmoid()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0] # according to pytorch doc for BERTPretrainedModel: (batch_size, sequence_length, hidden_size)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output) # according to pytorch doc for BERTPretrainedModel: (batch_size, hidden_size)
        logits = self.classifier(pooled_output)
        probs = self.sigm(logits)

        outputs = (logits, probs,) + (sequence_output, pooled_output,) + outputs[2:]  # + outputs[2:] adds hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, probs, sequence_ouput, pooled_output, # (hidden_states), (attentions)


def save_bert_model(model_to_save, model_dir, identifier):
    ''' Save finetuned (finished or intermediate) BERT model to a checkpoint. '''
    output_dir = os.path.join(model_dir, identifier)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(output_dir, "config.json")

    model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)


class BertWrapper:
    def __init__(self, cp_dir, n_eps, n_train_batches, load_from_path=0,
                 bert_model='bert-base-cased', cache_dir='models/cache', num_labels=2,
                 bert_lr=2e-6, warmup_proportion=0.1, seed_val=None):


        self.warmup_proportion = warmup_proportion
        self.device, self.use_cuda = get_torch_device()
        self.cache_dir = cache_dir
        self.cp_dir = cp_dir
        self.num_labels = num_labels

        self.model = self.load_model(bert_model=bert_model, load_from_path=load_from_path)
        self.model.to(self.device)
        if self.use_cuda:
            self.model.cuda()

        # set criterion, optim and scheduler
        self.criterion = CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=bert_lr, eps=1e-8)
        num_train_optimization_steps = n_train_batches * n_eps
        num_train_warmup_steps = int(self.warmup_proportion * num_train_optimization_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_train_warmup_steps,
                                                         num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
        #stepsize = int(n_train_batches/2)
        #self.scheduler = CyclicLR(self.optimizer, base_lr=bert_lr, max_lr=bert_lr*3,
        #                          step_size_up=stepsize, cycle_momentum=False)

    def train_on_batch(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        inputs, labels = batch[:-1], batch[-1]
        input_ids, input_mask, _, _ = inputs

        self.model.zero_grad()
        print('---')
        print(input_ids, input_mask)
        outputs = self.model(input_ids, input_mask, labels=labels)
        (loss), logits, probs, sequence_ouput, pooled_output = outputs
        print(logits, probs)
        print('---')
        loss = self.criterion(logits.view(-1, 2), labels.view(-1))
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def predict(self, batches):
        self.model.eval()

        y_pred = []
        sum_loss = 0
        embeddings = []
        for step, batch in enumerate(batches):
            batch = tuple(t.to(self.device) for t in batch)
            inputs, labels = batch[:-1], batch[-1]
            input_ids, input_mask, _, _ = inputs

            with torch.no_grad():
                outputs = self.model(input_ids, input_mask, labels=None)
                logits, probs, sequence_output, pooled_output, hidden_states = outputs
                loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
                probs = probs.detach().cpu().numpy()

                embedding = list(pooled_output.detach().cpu().numpy())
                embeddings.append(embedding)

            if len(y_pred) == 0:
                y_pred.append(probs)
            else:
                y_pred[0] = np.append(y_pred[0], probs, axis=0)
            sum_loss += loss.item()

        y_pred = np.argmax(y_pred[0], axis=1)
        return y_pred, sum_loss / len(batches), embeddings

    def get_embedding_output(self, batch, emb_type):
        batch = tuple(t.to(self.device) for t in batch)
        _, _, input_ids, input_mask, label_ids = batch

        with torch.no_grad():
            outputs = self.model(input_ids, input_mask, labels=None)
            logits, probs, sequence_output, pooled_output = outputs

            if emb_type == 'avbert':
                return sequence_output.mean(axis=1)

            elif emb_type == 'poolbert':
                return pooled_output

    def get_embeddings(self, batches, emb_type, model_path=''):
        if model_path:
            self.load_model(load_from_path=model_path)

        self.model.eval()
        embeddings = []
        for step, batch in enumerate(batches):
            emb_output = self.get_embedding_output(batch, emb_type)

            if self.use_cuda:
                emb_output = list(emb_output.detach().cpu().numpy()) # .detach().cpu() necessary here on gpu
            else:
                emb_output = list(emb_output.numpy())
            embeddings.append(emb_output)
        return embeddings

    def save_model(self, name):
        """
        Save bert model.
        :param model_dir: usually models/bert_for_embed/etc.
        :param name: usually number of current epoch
        """
        model_to_save = self.model

        output_dir = os.path.join(self.cp_dir, name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        output_config_file = os.path.join(output_dir, "config.json")

        model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def load_model(self, load_from_path=None, bert_model='bert-base-cased'):
        if not load_from_path:
            return BertForSequenceClassification.from_pretrained(bert_model, cache_dir=self.cache_dir,
                                                                 num_labels=self.num_labels,
                                                                 output_hidden_states=False,
                                                                 output_attentions=False)
        elif load_from_path:
            return BertForSequenceClassification.from_pretrained(load_from_path, num_labels=self.num_labels,
                                                                 output_hidden_states=False,
                                                                 output_attentions=False)
        #elif load_from_ep:
        #    load_dir = os.path.join(self.cp_dir, load_from_ep)
        #    return BertForSequenceClassification.from_pretrained(load_dir, num_labels=self.num_labels,
        #                                                         output_hidden_states=False,
        #                                                         output_attentions=False)

