import torch
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
import os, math, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
#plt.switch_backend('agg')
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import pandas as pd
import re

def standardise_id(basil_id):
    if not basil_id[1].isdigit():
        basil_id = '0' + basil_id
    if not basil_id[-2].isdigit():
        basil_id = basil_id[:-1] + '0' + basil_id[-1]
    return basil_id.lower()


def to_tensor(features):
    example_ids = [f.my_id for f in features]
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(input_ids, input_mask, label_ids)
    return example_ids, data, label_ids  # example_ids, input_ids, input_mask, segment_ids, label_ids


def to_tensors(split=None, features=None, device=None, article_wise=False):
    """ Tmp. """

    # to array if needed
    if features:
        token_ids = [f.input_ids for f in features]
        token_mask = [f.input_mask for f in features]
        labels = [f.label_id for f in features]
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        token_mask = torch.tensor(token_mask, dtype=torch.long, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        ids = [f.my_id for f in features]
        segment_ids = [f.segment_ids for f in features]
        contexts_ph = torch.tensor(segment_ids, dtype=torch.long, device=device)
        positions_ph = labels

        if features:
            return ids, TensorDataset(token_ids, token_mask, labels), labels
            #return TensorDataset(token_ids, token_mask, contexts_ph, positions_ph, labels)

    else:
        if article_wise:
            token_ids = np.zeros((300, 76, 122)) # n article * doc len * sent len
            token_mask = np.zeros((300, 76, 122)) # n article * doc len * sent len
            labels = np.zeros((300, 76, 1)) # n article * doc len * sent len

            art_i = 0
            for n, gr in split.groupby(['story', 'source']):
                for sent_i, (_, r) in enumerate(gr.iterrows()):
                    token_ids[art_i, sent_i] = np.asarray(r.token_ids)
                    token_mask[art_i, sent_i] = np.asarray(r.token_mask)
                art_i += 1

            token_ids = token_ids[:art_i]
            token_mask = token_mask[:art_i]
            labels = labels[:art_i]

            token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            token_mask = torch.tensor(token_mask, dtype=torch.long, device=device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            return TensorDataset(token_ids, token_mask, labels)

        else:

            art_contexts = np.array([list(el) for el in split.art_context_doc_num.values])
            cov1_contexts = np.array([list(el) for el in split.cov1_context_doc_num.values])
            cov2_contexts = np.array([list(el) for el in split.cov2_context_doc_num.values])
            token_ids = [list(el) for el in split.token_ids.values]
            token_mask = [list(el) for el in split.token_mask.values]

            art_contexts = torch.tensor(art_contexts, dtype=torch.long, device=device)
            cov1_contexts = torch.tensor(cov1_contexts, dtype=torch.long, device=device)
            cov2_contexts = torch.tensor(cov2_contexts, dtype=torch.long, device=device)
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            token_mask = torch.tensor(token_mask, dtype=torch.long, device=device)
            positions = torch.tensor(split.position.to_numpy(), dtype=torch.long, device=device)
            quartiles = torch.tensor(split.quartile.to_numpy(), dtype=torch.long, device=device)
            srcs = torch.tensor(split.src_num.to_numpy(), dtype=torch.long, device=device)
            labels = torch.tensor(split.label.to_numpy(), dtype=torch.long, device=device)
            # return TensorDataset(token_ids, token_mask, contexts, positions, labels)
            return TensorDataset(token_ids, token_mask, art_contexts, cov1_contexts, cov2_contexts, positions, quartiles, srcs, labels)


def to_batches(tensors, batch_size, sampler):
    ''' Creates dataloader with input divided into batches. '''
    if sampler == 'random':
        sampler = RandomSampler(tensors)
    elif sampler == 'sequential':
        sampler = SequentialSampler(tensors) #RandomSampler(tensors)
    loader = DataLoader(tensors, sampler=sampler, batch_size=batch_size)
    return loader

'''
def to_tensor_for_bert(features, OUTPUT_MODE):
    example_ids = [f.my_id for f in features]
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if OUTPUT_MODE == "classification":
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif OUTPUT_MODE == "regression":
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    return example_ids, data, label_ids  # example_ids, input_ids, input_mask, segment_ids, label_ids
'''


def indexesFromSentence(lang, sentence, EOS_token):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return indexes


def tensorFromSentence(lang, sentence, EOS_token, device):
    indexes = indexesFromSentence(lang, sentence, EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def get_torch_device():
    use_cuda = False
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        use_cuda = True

        #print('There are %d GPU(s) available.' % torch.cuda.device_count())
        #print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        #print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device, use_cuda


def plot_scores(losses):
    tr_scores, dev_scores = zip(*losses)
    # print('debug loss plotting:')
    # print(tr_scores, dev_scores)
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(tr_scores)
    plt.plot(dev_scores)
    plt.legend(('train', 'dev'), loc='upper right')
    return plt

'''
def showPlot(points):
    # PLOTTING NOT CURRENTLY FUNCTIONING
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def asMinutes(s): # other option for formatting time
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
'''


def format_runtime(runtime):
    min = int(runtime // 60)
    sec = int(runtime % 60)
    return f'{min}m:{sec}s'


def format_checkpoint_filepath(cp_dir, bertcam=None, hidden_size='NA', epoch_number=None):
    if not epoch_number:
        print("Give epoch number to checkpoint name")
    cp_fn = f'{bertcam}_hidden{hidden_size}_lastepoch{epoch_number}.model'
    return os.path.join(cp_dir, cp_fn)


def standardise_id(basil_id):
    if not basil_id[1].isdigit():
        basil_id = '0' + basil_id
    if not basil_id[-2].isdigit():
        basil_id = basil_id[:-1] + '0' + basil_id[-1]
    return basil_id.lower()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, my_id, input_ids, input_mask, segment_ids, label_id):
        self.my_id = my_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

