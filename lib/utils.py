import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
import numpy as np
import re


def clean_mean(df, grby='', set_type=''):
    """
    Helps with computing ML results by selecting the set type you want
    and computing rounded mean values on the grouping that you want.
    :param df: input dataframe with results (prec, rec, f1)
    :param grby: groupings, e.g. 'model' or 'seed'
    :param set_type: train, dev or test
    :return: means in an easily readible format
    """
    mets = ['f1']
    if set_type:
        tmp_df = df[df.set_type == set_type]
    else:
        tmp_df = df
    return tmp_df.groupby(grby)[mets].mean().round(2)


def standardise_id(basil_id):
    """
    Original basil ids are not all the same length.
    This functions inserts zeros in front of single-digit segments of the id to remedy this.
    :param basil_id: e.g. 1fox1, 2HPO3
    :return: standardized id, e.g. 01fox01, 02hpo03
    """
    if not basil_id[1].isdigit():
        basil_id = '0' + basil_id
    if not basil_id[-2].isdigit():
        basil_id = basil_id[:-1] + '0' + basil_id[-1]
    return basil_id.lower()


def plm_feats_to_tensors(features):
    """
    Converts roberta and bert-adapted features to tensors for batching
    """
    example_ids = [f.my_id for f in features]
    label_ids = [np.asarray(f.label_id) for f in features]
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    tensor_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(input_ids, input_mask, tensor_label_ids)
    return example_ids, data, label_ids  # example_ids, input_ids, input_mask, segment_ids, label_ids


def to_tensors(split=None, device=None):
    """
    Converts CIM features to tensors for batching
    """
    art_contexts = np.array([list(el) for el in split.art_context_doc_num.values])
    ev1_contexts = np.array([list(el) for el in split.ev1_context_doc_num.values])
    ev2_contexts = np.array([list(el) for el in split.ev2_context_doc_num.values])
    token_ids = [list(el) for el in split.token_ids.values]
    token_mask = [list(el) for el in split.token_mask.values]

    art_contexts = torch.tensor(art_contexts, dtype=torch.long, device=device)
    ev1_contexts = torch.tensor(ev1_contexts, dtype=torch.long, device=device)
    ev2_contexts = torch.tensor(ev2_contexts, dtype=torch.long, device=device)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    token_mask = torch.tensor(token_mask, dtype=torch.long, device=device)
    positions = torch.tensor(split.position.to_numpy(), dtype=torch.long, device=device)
    #quartiles = torch.tensor(split.quartile.to_numpy(), dtype=torch.long, device=device)
    srcs = torch.tensor(split.src_num.to_numpy(), dtype=torch.long, device=device)
    labels = torch.tensor(split.label.to_numpy(), dtype=torch.long, device=device)
    # return TensorDataset(token_ids, token_mask, contexts, positions, labels)
    return TensorDataset(token_ids, token_mask, art_contexts, ev1_contexts, ev2_contexts, positions, srcs, labels)


def to_batches(tensors, batch_size, sampler):
    """ Creates dataloader with input divided into batches. """
    if sampler == 'random':
        sampler = RandomSampler(tensors)
    elif sampler == 'sequential':
        sampler = SequentialSampler(tensors) #RandomSampler(tensors)
    loader = DataLoader(tensors, sampler=sampler, batch_size=batch_size)
    return loader


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
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device, use_cuda


def arrays_in_series(series):
    flat = []
    for el in series:
        el = el.strip('[ ]')
        el = re.sub('  ',' ',el)
        try:
            el = tuple(map(int, el.split(' ')))
        except:
            print('Unusual input in series:', el)
            print(el.split(' '))
        flat.extend(el)
    return np.asarray(flat)


def format_runtime(runtime):
    min = int(runtime // 60)
    sec = int(runtime % 60)
    return f'{min}m:{sec}s'


def standardise_id(basil_id):
    if not basil_id[1].isdigit():
        basil_id = '0' + basil_id
    if not basil_id[-2].isdigit():
        basil_id = basil_id[:-1] + '0' + basil_id[-1]
    return basil_id.lower()


class InputFeatures(object):
    """ Needed for loading features from pickle that were packaged with this class."""

    def __init__(self, my_id, input_ids, input_mask, segment_ids, label_id):
        self.my_id = my_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

