import torch
from torch import nn
from torch.autograd import Variable
from transformers.optimization import AdamW
from lib.Eval import my_eval
from lib.utils import format_runtime, get_torch_device
import os, time
import numpy as np

from torch.nn import CrossEntropyLoss, Embedding, Dropout, Linear, Sigmoid, LSTM


class Classifier:
    """
    Generic Classifier that performs recurring machine learning tasks
    """
    def __init__(self, model, logger, name, n_eps=10, patience=3, printing=100, load_from_ep=None):
        self.wrapper = model
        self.n_epochs = n_eps
        self.logger = logger
        self.patience = patience
        self.model_name = name
        self.print_every = printing

        # load
        self.epochs = range(n_eps)
        if load_from_ep:
            self.n_epochs += load_from_ep
            self.epochs = range(load_from_ep, self.n_epochs)
        else:
            self.epochs = range(1, self.n_epochs+1)

        # empty now and set during or after training
        self.train_time = 0
        self.prev_val_f1 = 0
        self.best_val_mets = {'f1':0}
        self.best_val_perf = ''
        self.full_patience = patience
        self.current_patience = self.full_patience
        self.test_mets = {}
        self.test_perf_string = ''
        self.cur_fold = ''
        self.best_model_loc = ''

    def train_epoch(self, train_batches):
        start = time.time()
        epoch_loss = 0
        for step, batch in enumerate(train_batches):

            loss = self.wrapper.train_on_batch(batch)
            epoch_loss += loss

            if (step > 0) & (step % self.print_every == 0):
                self.logger.info(f' > Step {step}/{len(train_batches)}: loss = {round(epoch_loss/step,4)}')

        av_epoch_loss = epoch_loss / len(train_batches)
        elapsed = format_runtime(time.time() - start)
        return av_epoch_loss, elapsed

    def update_patience(self, val_f1):
        # if an improvement happens, we have full patience, if no improvement happens
        # patience goes down, if patience reaches zero, we stop training
        if val_f1 > self.prev_val_f1:
            self.current_patience = self.full_patience
        else:
            self.current_patience -= 1
        self.prev_val_f1 = val_f1

    def unpack_fold(self, fold, voter_i):
        self.cur_fold = fold['name']
        tr_bs, tr_lbs = fold['train_batches'][voter_i], fold['train'][voter_i].label
        dev_bs, dev_lbs = fold['dev_batches'][voter_i], fold['dev'][voter_i].label
        return tr_bs, tr_lbs, dev_bs, dev_lbs

    def validate_after_epoch(self, ep, elapsed, fold, voter_i):
        ep_name = self.model_name + f"_ep{ep}"

        tr_bs, tr_lbs, dev_bs, dev_lbs = self.unpack_fold(fold, voter_i=voter_i)

        tr_preds, tr_loss, _, _ = self.wrapper.predict(tr_bs)
        tr_mets, tr_perf = my_eval(tr_lbs, tr_preds, set_type='train', av_loss=tr_loss, name="")

        val_preds, val_loss, _, _ = self.wrapper.predict(dev_bs)
        val_mets, val_perf = my_eval(dev_lbs, val_preds, set_type='dev', av_loss=val_loss, name="")

        best_log = ''
        if val_mets['f1'] > self.best_val_mets['f1']:
            self.best_val_mets = val_mets
            self.best_val_mets['epoch'] = ep
            self.best_model_loc = ep_name
            self.wrapper.save_model(self.model_name)
            best_log = '(HIGH SCORE)'

        self.logger.info(f" Ep {ep} ({self.model_name.replace('_', '')}): "
                         f"{tr_perf} | {val_perf} {best_log}")

        return tr_mets, tr_perf, val_mets, val_perf

    def train_all_epochs(self, fold, voter_i):
        tr_bs, tr_lbs, dev_bs, dev_lbs = self.unpack_fold(fold, voter_i)
        train_start = time.time()
        losses = []

        if self.model_name == 'BERT':
            elapsed = format_runtime(time.time() - train_start)
            tr_mets, tr_perf, val_mets, val_perf = self.validate_after_epoch(-1, elapsed, fold, voter_i)
            losses.append((tr_mets['loss'], val_mets['loss']))

        for ep in self.epochs:
            self.wrapper.model.train()

            av_tr_loss, ep_elapsed = self.train_epoch(tr_bs)

            tr_mets, tr_perf, val_mets, val_perf = self.validate_after_epoch(ep, ep_elapsed, fold, voter_i)
            losses.append((av_tr_loss, val_mets['loss']))

            self.update_patience(val_mets['f1'])

            if (not self.current_patience > 0) & (val_mets['f1'] > 0.20):
                self.logger.info(" > Stopping training.")
                break

        eps_elapsed = format_runtime(time.time() - train_start)
        return eps_elapsed, losses

    def test_model(self, fold, name):
        preds, test_loss, _, _ = self.wrapper.predict(fold['test_batches'])
        test_mets, test_perf = my_eval(fold['test'].label, preds, name=name, set_type='test', av_loss=test_loss)
        return test_mets, test_perf

    def train_on_fold(self, fold, voter_i):
        self.cur_fold = fold['name']
        train_elapsed, losses = self.train_all_epochs(fold, voter_i)
        self.train_time = train_elapsed

        # plot learning curve
        #loss_plt = plot_scores(losses)
        #loss_plt.savefig(self.fig_dir + f'/{self.model_name}_trainval_loss.png', bbox_inches='tight')

        # test_model
        if self.best_model_loc:
            self.wrapper.load_model(self.model_name)
            self.logger.info(f'Loaded best model from {self.best_model_loc}')

            name = self.model_name + f"_TEST_{self.n_epochs}"

            # test_mets, test_perf = self.test_model(fold, name)
            preds, test_loss, _, _ = self.wrapper.predict(fold['test_batches'])
            test_mets, test_perf = my_eval(fold['test'].label, preds, name=name, set_type='test', av_loss=test_loss)

            self.logger.info(f' FINISHED training {name} (took {self.train_time})')
            self.logger.info(f" {test_mets}")
        else:
            test_mets = None

        return self.best_val_mets, test_mets, preds

    def produce_preds(self, fold, model_name):
        if not model_name:
            model_name = self.model_name
        self.wrapper.load_model(model_name)
        preds, _, _, losses = self.wrapper.predict(fold['test_batches'])
        return preds, losses


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention from: https://bastings.github.io/annotated_encoder_decoder/"""

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        query = query.unsqueeze(1)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = nn.functional.softmax(scores, dim=-1)
        self.alphas = alphas

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


class ContextAwareModel(nn.Module):
    """
    Model that uses BiLSTMs and classification of hidden representation of token at target index to do context-aware prediction.
    :param input_size: length of input sequences (= documents)
    :param hidden_size: size of hidden layer
    :param weights_matrix: matrix of embeddings of size vocab_size * embedding dimension
    :param cim_type: cim or cim*
    :param context: art (article) or ev (event)
    :param pos_dim: not used in the paper: dimension of embedding for position of target sentence in document
    :param src_dim: dimension of embedding for news source (publisher)
    :param nr_pos_bins: number of categories for position in article (e.g. quartiles)
    :param nr_srcs: number of sources (publishers)

    """
    def __init__(self, input_size, hidden_size, bilstm_layers, weights_matrix, cim_type, device, context='art',
                 pos_dim=100, src_dim=100, nr_pos_bins=4, nr_srcs=3):
        super(ContextAwareModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size # + pos_dim + src_dim
        self.bilstm_layers = bilstm_layers
        self.device = device
        self.cim_type = cim_type
        self.context = context

        # Store pretrained embeddings to use as representations of sentences
        self.weights_matrix = torch.tensor(weights_matrix, dtype=torch.float, device=self.device)
        self.embedding = Embedding.from_pretrained(self.weights_matrix)
        self.embedding_pos = Embedding(nr_pos_bins, pos_dim)  # option to embed position of target sentence in article
        self.embedding_src = Embedding(nr_srcs, src_dim)
        self.emb_size = weights_matrix.shape[1]

        # Initialise LSTMS for article and event context
        self.lstm_art = LSTM(self.input_size, self.hidden_size, num_layers=self.bilstm_layers, bidirectional=True, dropout=0.2)
        self.lstm_ev1 = LSTM(self.input_size, self.hidden_size, num_layers=self.bilstm_layers, bidirectional=True, dropout=0.2)
        self.lstm_ev2 = LSTM(self.input_size, self.hidden_size, num_layers=self.bilstm_layers, bidirectional=True, dropout=0.2)

        # Attention-related attributes
        # self.attention = BahdanauAttention(self.hidden_size, key_size=self.hidden_size * 2, query_size=self.emb_size)
        # self.rob_squeezer = nn.Linear(self.emb_size, self.hidden_size)

        self.dropout = Dropout(0.6)
        self.num_labels = 2
        self.pad_index = 0

        if self.context == 'art':
            self.context_rep_dim = self.emb_size + self.hidden_size * 2  # size of target sentences + 1 article
        else:
            self.context_rep_dim = self.emb_size + self.hidden_size * 6  # size of target sentences + 3 articles

        if self.cim_type == 'cim*':
            self.context_rep_dim += src_dim  #  add representation of source

        self.half_context_rep_dim = int(self.context_rep_dim*0.5)
        self.dense = nn.Linear(self.context_rep_dim, self.half_context_rep_dim)

        if self.cim_type == 'cnm':
            # optional Context Naive setting
            self.classifier = Linear(self.emb_size, self.num_labels)
        else:
            self.classifier = Linear(self.half_context_rep_dim, self.num_labels) # + self.emb_size + src_dim, 2) #

        self.sigm = Sigmoid()

    def forward(self, inputs):
        """
        Forward pass.
        :param input_tensor: batchsize * seq_length
        :param target_idx: batchsize, specifies which token is to be classified
        :return: sigmoid output of size batchsize
        """

        # inputs
        # token_ids, token_mask, contexts, positions = inputs
        token_ids, token_mask, article, ev1, ev2, positions, quartiles, srcs = inputs

        # shapes and sizes
        batch_size = inputs[0].shape[0]
        sen_len = token_ids.shape[1]
        doc_len = article.shape[1]
        seq_len = doc_len

        # init containers for outputs
        rep_dimension = self.emb_size if self.cim_type == 'cnm' else self.hidden_size * 2
        art_representations = torch.zeros(batch_size, seq_len, rep_dimension, device=self.device)

        if self.context != 'art':
            ev1_representations = torch.zeros(batch_size, seq_len, rep_dimension, device=self.device)
            ev2_representations = torch.zeros(batch_size, seq_len, rep_dimension, device=self.device)

        target_sent_reps = torch.zeros(batch_size, self.emb_size, device=self.device)

        if self.cim_type == 'cnm':
            # optional Context Naive setting
            target_sent_reps = torch.zeros(batch_size, rep_dimension, device=self.device)
            for item, position in enumerate(positions):
                target_sent_reps[item] = self.embedding(article[item, position]).view(1, -1)

        else:
            for item, position in enumerate(positions):
                # target_hid = sentence_representations[item, position].view(1, -1)
                target_roberta = self.embedding(article[item, position]).view(1, -1)
                # target_sent_reps[item] = torch.cat((target_hid, target_roberta), dim=1)
                # target_sent_reps[item] = target_hid
                target_sent_reps[item] = target_roberta

            embedded_pos = self.embedding_pos(quartiles)  # old line for experimenting with embedding position
            embedded_src = self.embedding_src(srcs)

            # embedding article

            hidden = self.init_hidden(batch_size)
            for seq_idx in range(article.shape[0]):
                embedded_sentence = self.embedding(article[:, seq_idx]).view(1, batch_size, -1)
                lstm_input = embedded_sentence # torch.cat((embedded_sentence, embedded_src), dim=-1)
                encoded, hidden = self.lstm_art(lstm_input, hidden)
                art_representations[:, seq_idx] = encoded
            final_article_reps = art_representations[:, -1, :]

            if self.cim_type == 'cnm':
                # embedding first event context piece
                hidden = self.init_hidden(batch_size)
                for seq_idx in range(article.shape[0]):
                    embedded_sentence = self.embedding(ev1[:, seq_idx]).view(1, batch_size, -1)
                    encoded, hidden = self.lstm_ev1(embedded_sentence, hidden)
                    ev1_representations[:, seq_idx] = encoded
                final_ev1_reps = ev1_representations[:, -1, :]

                # embedding second event context piece
                hidden = self.init_hidden(batch_size)
                for seq_idx in range(article.shape[0]):
                    embedded_sentence = self.embedding(ev2[:, seq_idx]).view(1, batch_size, -1)
                    encoded, hidden = self.lstm_ev2(embedded_sentence, hidden)
                    ev2_representations[:, seq_idx] = encoded
                final_ev2_reps = ev2_representations[:, -1, :]

                context_reps = torch.cat((final_article_reps, final_ev1_reps, final_ev2_reps), dim=-1)
            else:
                context_reps = final_article_reps

            # Attention-related processing
            # target_sent_reps = self.rob_squeezer(target_sent_reps)
            # query = target_sent_reps.unsqueeze(1)
            # proj_key = self.attention.key_layer(sentence_representations)
            # mask = (contexts != self.pad_index).unsqueeze(-2)

            if self.cim_type == 'cim':
                context_and_target_rep = torch.cat((target_sent_reps, context_reps), dim=-1)
                # context_and_target_rep, attn_probs = self.attention(query=target_sent_reps, proj_key=proj_key,
                #                                         value=sentence_representations, mask=mask)
                # context_and_target_rep = torch.cat((target_sent_reps, context_and_target_rep), dim=-1)
            elif self.cim_type == 'cim*':
                context_and_target_rep = torch.cat((target_sent_reps, context_reps, embedded_src), dim=-1)

        # Linear classification
        features = self.dropout(context_and_target_rep)
        features = self.dense(features)
        features = torch.tanh(features)
        features = self.dropout(features)
        logits = self.classifier(features)
        probs = self.sigm(logits)

        return logits, probs, target_sent_reps

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.bilstm_layers * 2, batch_size, self.hidden_size, device=self.device)
        cell = torch.zeros(self.bilstm_layers * 2, batch_size, self.hidden_size, device=self.device)
        return Variable(hidden), Variable(cell)


class CIMClassifier():
    def __init__(self, emb_dim=768, hid_size=32, layers=1, weights_mat=None, tr_labs=None,
                 b_size=24, cp_dir='models/checkpoints/cim', lr=0.001, start_epoch=0, patience=3,
                 step=1, gamma=0.75, n_eps=10, cim_type='cim', context='art'):
        self.start_epoch = start_epoch
        self.cp_dir = cp_dir
        self.device, self.use_cuda = get_torch_device()

        self.emb_dim = emb_dim
        self.hidden_size = hid_size
        self.batch_size = b_size
        if cim_type == 'cim':
            self.criterion = CrossEntropyLoss(weight=torch.tensor([.20, .80], device=self.device), reduction='sum')  # could be made to depend on classweight which should be set on input
        else:
            self.criterion = CrossEntropyLoss(weight=torch.tensor([.25, .75], device=self.device), reduction='sum')  # could be made to depend on classweight which should be set on input

        # self.criterion = NLLLoss(weight=torch.tensor([.15, .85], device=self.device))
        # set criterion on input
        # n_pos = len([l for l in tr_labs if l == 1])
        # class_weight = 1 - (n_pos / len(tr_labs))
        # print(class_weight)
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([.85], reduction='sum', dtype=torch.float, device=self.device))

        if start_epoch > 0:
            self.model = self.load_model()
        else:
            self.model = ContextAwareModel(input_size=self.emb_dim, hidden_size=self.hidden_size,
                                           bilstm_layers=layers, weights_matrix=weights_mat,
                                           device=self.device, cim_type=cim_type, context=context)
        self.model = self.model.to(self.device)
        if self.use_cuda: self.model.cuda()

        # empty now and set during or after training
        self.train_time = 0
        self.prev_val_f1 = 0
        self.cp_name = None  # depends on split type and current fold
        self.full_patience = patience
        self.current_patience = self.full_patience
        self.test_perf = []
        self.test_perf_string = ''

        # set optimizer
        nr_train_instances = len(tr_labs)
        nr_train_batches = int(nr_train_instances / b_size)
        half_tr_bs = int(nr_train_instances/2)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, eps=1e-8)

        # set scheduler if desired
        # self.scheduler = lr_scheduler.CyclicLR(self.optimizer, base_lr=lr, step_size_up=half_tr_bs,
        #                                       cycle_momentum=False, max_lr=lr * 30)
        num_train_warmup_steps = int(0.1 * (nr_train_batches * n_eps)) # warmup_proportion
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_train_warmup_steps,
        # num_training_steps=num_train_optimization_steps)

    def load_model(self, name):
        cpfp = os.path.join(self.cp_dir, name)
        cp = torch.load(cpfp, map_location=torch.device('cpu'))
        model = cp['model']
        model.load_state_dict(cp['state_dict'])
        self.model = model
        self.model.to(self.device)
        if self.use_cuda: self.model.cuda()
        return model

    def train_on_batch(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        inputs, labels = batch[:-1], batch[-1]

        self.model.zero_grad()
        logits, probs, _ = self.model(inputs)
        loss = self.criterion(logits.view(-1, 2), labels.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        #self.scheduler.step()
        return loss.item()

    def save_model(self, name):
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        cpfp = os.path.join(self.cp_dir, name)
        torch.save(checkpoint, cpfp)

    def predict(self, batches):
        self.model.eval()

        y_pred = []
        losses = []
        sum_loss = 0
        embeddings = []
        for step, batch in enumerate(batches):
            batch = tuple(t.to(self.device) for t in batch)
            inputs, labels = batch[:-1], batch[-1]

            with torch.no_grad():
                logits, probs, sentence_representation = self.model(inputs)
                loss = self.criterion(logits.view(-1, 2), labels.view(-1))
                # loss = self.criterion(logits.squeeze(), labels)

                embedding = list(sentence_representation.detach().cpu().numpy())
                embeddings.append(embedding)

            loss = loss.detach().cpu().numpy()  # probs.shape: batchsize * num_classes
            probs = probs.detach().cpu().numpy()  # probs.shape: batchsize * num_classes

            losses.append(loss)

            if len(y_pred) == 0:
                y_pred = probs
            else:
                y_pred = np.append(y_pred, probs, axis=0)


                # convert to predictions
                # #preds = [1 if output > 0.5 else 0 for output in sigm_output]
                #y_pred.extend(preds)

            sum_loss += loss.item()

        y_pred = y_pred.squeeze()
        y_pred = np.argmax(y_pred, axis=1)
        # y_pred = [0 if el < 0.5 else 1 for el in y_pred]
        self.model.train()
        return y_pred, sum_loss / len(batches), embeddings, losses
