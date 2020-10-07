import torch
from torch import nn
from torch.autograd import Variable
from transformers.optimization import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
from lib.evaluate.StandardEval import my_eval
from lib.utils import indexesFromSentence, format_runtime, format_checkpoint_filepath, get_torch_device
import os, time

SOS_token = 0
EOS_token = 1

"""
Based on: NLP From Scratch: Translation with a Sequence to Sequence Network and Attention
*******************************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_
"""

'''
class FFLayer(nn.Module):
    def __init__(self, hidden_size):
        super(FFLayer, self).__init__()
        self.hidden_size = hidden_size
        self.out = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Sigmoid())

    def forward(self, input):
        return self.out(input)
'''

class ContextAwareModel(nn.Module):
    def __init__(self, input_size, hidden_size, weights_matrix, device):
        super(ContextAwareModel, self).__init__()
        '''
        :param input_size = sequence length / max length
        :param hidden_size = typically 32
        :param weights_matrix = vocab_size * embedding dimension (typically [7987, 768])
        Structure of the model: 
        i) enter pretrained embeddings (USE, avbert or sbert) (= weights_matrix)
        ii) bi-LSTM pass for each token (=sentence) in sequence (=article), storing the hidden state corresponding to each token  
        iii) return as output Linear + Sigmoid applied to hidden state of only the token (= sentence) in question
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.weights_matrix = torch.tensor(weights_matrix, dtype=torch.float, device=self.device)
        self.embedding = nn.Embedding.from_pretrained(self.weights_matrix)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(self.hidden_size * 2, 1), nn.Sigmoid())

    def forward(self, input_tensor, target_idx, max_length):
        '''
        :param input_tensor: batchsize * seq_length
        :param target_idx: batchsize
        :param max_length: max sequence length, 77 in case of basil
        :return:
        '''
        batch_size = input_tensor.shape[0]
        input_length = input_tensor.shape[1]

        encoder_outputs = torch.zeros(input_length, batch_size, self.hidden_size * 2, device=self.device)
        hidden = self.initHidden(batch_size)

        # loop through input
        for ei in range(input_length):
            # get sentence embedding for that item
            embedded = self.embedding(input_tensor[:,ei]).view(1, batch_size, -1)
            # feed hidden of previous token/item, store in hidden again
            output, hidden = self.lstm(embedded, hidden) # output has shape 1 (for the token in question) * batch_size * hiddenx2
            encoder_outputs[ei] = output[0]

        target_encoder_output = torch.zeros(batch_size, 1, self.hidden_size * 2, device=self.device)
        for item in range(batch_size):
            my_idx = target_idx[item]
            target_encoder_output[item] = encoder_outputs[my_idx,item,:]

        output = self.out(target_encoder_output)  # sigmoid function that returns batch_size * 1
        return output

    def initHidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, self.hidden_size, device=self.device)
        cell = torch.zeros(2, batch_size, self.hidden_size, device=self.device)
        return Variable(hidden), Variable(cell)


class ContextAwareClassifier():
    def __init__(self, input_lang, dev, test, logger=None,
                 emb_dim=768, hidden_size=32, weights_matrix=None,
                 batch_size=24, cp_dir='models/checkpoints/cim',
                 learning_rate=0.001, start_epoch=0, step_size=1, gamma=0.75):
        self.start_epoch = start_epoch
        self.cp_dir = cp_dir
        self.best_cp_dir = os.path.join(cp_dir, 'best')
        self.device, self.USE_CUDA = get_torch_device()
        self.logger = logger

        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_lang = input_lang
        self.max_length = input_lang.max_len
        self.criterion = None # depends on classweight which should be set on input

        if start_epoch > 0:
            self.model = self.load_model_from_checkpoint()
        else:
            self.model = ContextAwareModel(input_size=self.emb_dim, hidden_size=self.hidden_size,
                                           weights_matrix=weights_matrix, device=self.device)
        self.model = self.model.to(self.device)
        if self.USE_CUDA: self.model.cuda()

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        self.scheduler = lr_scheduler.CyclicLR(self.optimizer, base_lr=learning_rate, max_lr=0.1) #StepLr: step_size=step_size, gamma=gamma)
        self.dev = dev
        self.test = test
        self.cp_name = None # depends on split type and current fold
        self.best_perf = {'ep': 0, 'val_f1': 30}

    def load_model_from_checkpoint(self):
        cpfp = format_checkpoint_filepath(self.cp_dir, self.hidden_size, epoch_number=self.start_epoch)
        self.logger.info('Loading model from', cpfp)
        start_checkpoint = torch.load(cpfp)
        model = start_checkpoint['model']
        model.load_state_dict(start_checkpoint['state_dict'])
        return model

    def to_tensor(self, triples):
        indexedsentences = [indexesFromSentence(self.input_lang, t[0], EOS_token) for t in triples]
        input_tensor = torch.tensor(indexedsentences, dtype=torch.long, device=self.device)
        target_label_tensor = torch.tensor([t[1] for t in triples], dtype=torch.float, device=self.device)
        idx = torch.tensor([t[2] for t in triples], dtype=torch.long, device=self.device)
        data = TensorDataset(input_tensor, target_label_tensor, idx)
        return data

    def train(self, input_tensor, target_label_tensor, target_idx):
        #self.model.zero_grad()
        self.optimizer.zero_grad()

        loss = 0
        output = self.model(input_tensor, target_idx, self.max_length)
        #print(output)
        #print(target_label_tensor)
        #print(output.shape, target_label_tensor.shape)
        loss += self.criterion(output, target_label_tensor)
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def save_checkpoint(self, cpdir, ep):
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        checkpoint_name = format_checkpoint_filepath(cpdir, self.hidden_size, epoch_number=ep)
        torch.save(checkpoint, checkpoint_name)

    def update_lr(self, best_ep, val_f1):
        self.best_perf['ep'] = best_ep
        self.best_perf['val_f1'] = val_f1
        self.scheduler.step()
        new_lr = self.scheduler.get_lr()
        self.logger.info('\t\t{} - Updated LR: {} for f1 = {}'.format(self.best_perf['ep'], new_lr, self.best_perf['val_f1']))
        val_performance = self.evaluate(self.dev, which='string')
        test_performance = self.evaluate(self.test, which='string')
        self.logger.info(f'\t\t\t Val performance: {val_performance}, Test performance: {test_performance}')

    def train_batches(self, fold, print_step_every):
        self.cp_name = fold['name']

        training_triples = self.to_tensor(fold['train'])
        train_sampler = RandomSampler(training_triples)
        train_dataloader = DataLoader(training_triples, sampler=train_sampler, batch_size=self.batch_size)

        nr_steps = len(train_dataloader)

        loss_total = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_tensor, target_label_tensor, target_idx = batch

            loss = self.train(input_tensor, target_label_tensor, target_idx)
            loss_total += loss

            if (step % print_step_every == 0) & (step > 0):
                update = f'\t\tFinished step {step}/{nr_steps} - loss: {loss}, lr: {self.scheduler.get_lr()}'
                self.logger.info(update)
        av_loss = loss_total / len(train_dataloader)

        return av_loss

    def train_epochs(self, fold, num_epochs, print_step_every, save_epoch_every):
        pos_freq = [t for t in fold['train'] if t[1] == 1]
        class_weight = 1 - (len(pos_freq) / len(fold['train']))
        self.criterion = nn.BCELoss(weight=torch.tensor(class_weight, dtype=torch.float, device=self.device))

        self.logger.info('Training...')
        total_loss = 0
        for ep in range(num_epochs):
            ep = self.start_epoch + ep
            start_time = time.time()

            epoch_av_loss = self.train_batches(fold, print_step_every)
            total_loss += epoch_av_loss

            elapsed = format_runtime(time.time()-start_time)
            self.scheduler.step()
            if (ep % save_epoch_every == 0) & (ep > 0):
                epochs_av_loss = total_loss / ep
                val_performance = self.evaluate(fold['dev'], which='string')
                update = f'\tEpoch {ep}/{num_epochs} (took {elapsed}): Av loss: {epoch_av_loss}, Val performance: {val_performance})'
                self.logger.info(update)
                self.save_checkpoint(self.cp_dir, ep=ep)
                self.decide_if_schedule_step(ep)

    '''
    def evaluate(self, test_triples, which='f1'):
        with torch.no_grad():
            y_pred = []
            y_true = [t[1] for t in test_triples]
            for tr in test_triples:
                input_tensor, _, idx = self.to_tensor([tr])
                output = self.model(input_tensor, idx, self.max_length)
                pred = 1 if output > 0.5 else 0
                y_pred.extend([pred])
            metrics, metrics_string = evaluation(y_true, y_pred)
            if which == 'f1':
                f1 = round(metrics[-1] * 100,2)
                return f1
            elif which == 'all':
                return metrics, metrics_string
    '''

    def predict(self, data):
        test_triples = self.to_tensor(data)
        test_sampler = SequentialSampler(test_triples)
        test_dataloader = DataLoader(test_triples, sampler=test_sampler, batch_size=self.batch_size)
        y_true = []
        y_pred = []
        for batch in test_dataloader:
            # get output for batch
            batch = tuple(t.to(self.device) for t in batch)
            inputs, labels = batch
            outputs = self.model(inputs, self.max_length)
            outputs = outputs.detach().cpu().numpy()

            # convert to predictions
            preds = [1 if output > 0.5 else 0 for output in outputs]
            y_true.extend([el for el in labels.detach().cpu().numpy()])
            y_pred.extend(preds)
        return y_true, y_pred

    def evaluate(self, data, which='f1'):
        y_true, y_pred = self.predict(data)
        metrics, metrics_df, metrics_string = my_eval('eval', y_true, y_pred, opmode='classification')
        f1 = round(metrics['f1'] * 100, 2)

        if which == 'all':
            return metrics, metrics_df, metrics_string
        elif which == 'f1':
            return f1
        elif which == 'string':
            return metrics_string


#_, USE_CUDA = get_torch_device()
#LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
#FloatTensor = torch.cuda.FLoatTensor if USE_CUDA else torch.FloatTensor
