from __future__ import absolute_import, division, print_function
from transformers import RobertaTokenizer, BertTokenizer
import pickle
from lib.handle_data.PreprocessForPLM import BinaryClassificationProcessor, InputFeatures
from lib.handle_data.PreprocessForPLM import convert_basil_for_plm_inputs, convert_example_to_bert_feature, convert_example_to_roberta_feature
from lib.handle_data.SplitData import split_input_for_plm
from lib.handle_data.BasilLoader import LoadBasil
import os, time
import argparse
import spacy
import pandas as pd


def preprocess_for_plm(rows, model):
    count = 0
    total = len(rows)
    features = []
    for row in rows:
        if model == 'bert':
            feats = convert_example_to_bert_feature(row)
        elif model == 'roberta':
            feats = convert_example_to_roberta_feature(row)
        features.append(feats)
        count += 1
    return features


def enforce_max_sent_per_example(sentences, max_sent_per_example, labels=None):
    """
    Splits examples with len(sentences) > self.max_sent_per_example into multiple smaller examples
    with len(sentences) <= self.max_sent_per_example.
    Recursively split the list of sentences into two halves until each half
    has len(sentences) < <= self.max_sent_per_example. The goal is to produce splits that are of almost
    equal size to avoid the scenario where all splits are of size
    self.max_sent_per_example then the last split is 1 or 2 sentences
    This will result into losing context around the edges of each examples.
    """
    if labels is not None:
        assert len(sentences) == len(labels)

    if len(sentences) > max_sent_per_example > 0:
        i = len(sentences) // 2
        l1 = enforce_max_sent_per_example(
                sentences[:i], max_sent_per_example, None if labels is None else labels[i:])
        l2 = enforce_max_sent_per_example(
                sentences[i:], max_sent_per_example, None if labels is None else labels[i:])
        return l1 + l2
    else:
        return [sentences]


def flatten_sequence(seq_rows, cls, pad, max_ex_len, max_sent_in_ex, window):
    flat_input_ids = []
    flat_labels = []
    #segment_ids = []

    for i, sent in enumerate(seq_rows):
        input_ids = remove_special(sent.input_ids, cls, pad)
        flat_input_ids.extend(input_ids)
        flat_labels.append(sent.label_id)

    pad_len = max_ex_len - len(flat_input_ids)
    mask = [1] * len(flat_input_ids) + [0] * pad_len
    flat_input_ids += [pad] * pad_len
    #segment_ids += [pad] * pad_len

    assert len(mask) == len(flat_input_ids)

    if window:
        max_sent_in_ex = max_sent_in_ex + 2

    lab_pad_len = max_sent_in_ex - len(flat_labels)
    flat_labels += [-1] * lab_pad_len

    assert len(flat_labels) == max_sent_in_ex

    return InputFeatures(my_id=None,
                         input_ids=flat_input_ids,
                         input_mask=mask,
                         segment_ids=[],
                         label_id=flat_labels)


def remove_special(x, cls=0, pad=1):
    return [el for el in x if el not in [cls, pad]]


def seps(x):
    #mask = x == 2
    return [el for el in x if el == 2]#x[mask]


def redistribute_feats(features, cls=0, pad=1, max_sent=10, max_len=None, window=True):
    """
    Takes rows of features (each row is sentence), and converts them to rows of multiple sentences.
    """

    empty_feature = InputFeatures(my_id=pad,
                                     input_ids=[],
                                     input_mask=[],
                                     segment_ids=[],
                                     label_id=-1)
    window_size = 1

    article_rows = {}

    for f in features:
        row = article_rows.setdefault(f.article, [])
        row.append(f)

    sequence_rows = []
    nr_sequences_agg = []
    for row in article_rows.values():
        row = sorted(row, key=lambda x: x.sent_id, reverse=False)

        if window:
            row = [empty_feature]*window_size + row + [empty_feature]*window_size

        sequences = enforce_max_sent_per_example(row, max_sent)
        nr_sequences = len(sequences)
        nr_sequences_agg.append(nr_sequences)

        for i, s in enumerate(sequences):
            if window:
                winseq = s.copy()
                if i != 0:
                    winstart = sequences[i-1][-window_size:]
                    winseq = winstart + winseq
                if i != nr_sequences-1:
                    winend = sequences[i+1][0:window_size]
                    winseq = winseq + winend
                sequence_rows.append(winseq)
            else:
                sequence_rows.append(s)

    # print('Av seq len of this fold:',sum(nr_sequences_agg) / len(article_rows))

    # help measure what the maxlen should be
    for row in sequence_rows:
        toks = [remove_special(f.input_ids, cls, pad) for f in row]
        exlen = sum([len(t) for t in toks])
        if exlen > max_len:
            max_len = exlen
            # print('MAX EX LEN of this fold:', max_len)

    finfeats = []
    for row in sequence_rows:
        ff = flatten_sequence(row, cls, pad, max_len, max_sent, window)
        finfeats.append(ff)
    return finfeats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-task', '--clf_task', type=str, default='sent_clf',
                        help='Choose classification task between: sent_clf|tok_clf|seq_sent_clf', )
    parser.add_argument('-plm', '--pretrained_lm', type=str, default='roberta',
                        help='BERT|RoBERTa')

    parser.add_argument('-seqlen', '--sequence_length', type=int, default=1,
                        help='If task is seq_sent_clf: Number of sentences per example')
    parser.add_argument('-w', '--windowed', action='store_true', default=False,
                        help='If task is seq_sent_clf: Choose Windowed SSC or not')

    args = parser.parse_args()

    CLF_TASK = args.clf_task
    PLM = args.pretrained_lm
    WINDOW = args.windowed
    MAX_EX_LEN = args.sequence_length
    MAX_DOC_LEN = 76
    MAX_SENT_LEN = 486

    if CLF_TASK == 'seq_sent_clf':
        if WINDOW:
            max_lens = {5: 398, 10: 543}
        else:
            max_lens = {5: 305, 10: 499}
        MAX_SEQ_LEN = max_lens[args.sequence_length]
    else:
        MAX_SEQ_LEN = 124

    ######
    # structure of project
    ######

    # Load DataFrame with BASIL instances and - selected - annotations
    basil = LoadBasil().load_basil_raw()
    basil.to_csv('data/basil.csv')
    basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')
    convert_basil_for_plm_inputs(basil, task=CLF_TASK, ofp=f'data/inputs/{CLF_TASK}/plm_basil.tsv')

    #############
    ### BASELINES
    #############

    if CLF_TASK == 'seq_sent_clf':
        if WINDOW:
            FEAT_DIR = f'data/inputs/{CLF_TASK}/windowed/ssc{MAX_EX_LEN}/'
        else:
            FEAT_DIR = f'data/inputs/{CLF_TASK}/non_windowed/ssc{MAX_EX_LEN}/'
    else:
        FEAT_DIR = f'data/inputs/{CLF_TASK}/features_for_{PLM}/'
    DATA_DIR = f'data/inputs/{CLF_TASK}'

    if not os.path.exists(FEAT_DIR):
        os.makedirs(FEAT_DIR)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    DATA_TSV_IFP = os.path.join(DATA_DIR, f"plm_basil.tsv")
    FEAT_OFP = os.path.join(FEAT_DIR, f"all_features.pkl")

    ###
    # load data
    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
    ###

    folds = split_input_for_plm(DATA_DIR, recreate=False, sv=99)
    NR_FOLDS = len(folds)

    ###
    # get relevant classes depending on task and model
    ###

    dataloader = BinaryClassificationProcessor()
    label_list = dataloader.get_labels(output_mode=CLF_TASK)  # [0, 1] for binary classification

    if CLF_TASK == 'tok_clf':
        spacy_tokenizer = spacy.load("en_core_web_sm")
        if PLM == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)
        elif PLM == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False, do_basic_tokenize=False)
        label_map = {label: i + 1 for i, label in enumerate(label_list)}

    elif CLF_TASK == 'sent_clf' or CLF_TASK == 'seq_sent_clf':
        spacy_tokenizer = None
        if PLM == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        elif PLM == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        label_map = {label: i for i, label in enumerate(label_list)}

    ###
    # write features for whole dataset
    # takes a while!
    ###

    FORCE = False
    if not os.path.exists(FEAT_OFP) or FORCE:
        examples = dataloader.get_examples(DATA_TSV_IFP, 'train', sep='\t')
        examples = [(ex, label_map, MAX_SEQ_LEN, tokenizer, spacy_tokenizer, CLF_TASK) for ex in examples if ex.text_a]
        features = preprocess_for_plm(examples, model=PLM)
        features_dict = {feat.my_id: feat for feat in features}

        with open(FEAT_OFP, "wb") as f:
            pickle.dump(features, f)
        time.sleep(15)
    else:
        with open(FEAT_OFP, "rb") as f:
           features = pickle.load(f)
           features_dict = {feat.my_id: feat for feat in features}

    print(f"Processed all {len(examples)} items")

    ###
    # write features in seperate files per fold
    ###

    for fold in folds:
        fold_name = fold['name']
        for set_type in ['train', 'dev', 'test']:
            infp = os.path.join(DATA_DIR, f"{fold_name}_{set_type}.tsv")
            FEAT_OFP = os.path.join(FEAT_DIR, f"{fold_name}_{set_type}_features.pkl")

            examples = dataloader.get_examples(infp, set_type, sep='\t')
            features = [features_dict[example.my_id] for example in examples if example.text_a]
            if CLF_TASK == 'seq_sent_clf':
                features = redistribute_feats(features, cls=0, pad=1, max_sent=MAX_EX_LEN, max_len=MAX_SEQ_LEN,
                                          window=WINDOW)
            # print(f"Processed fold {fold_name} {set_type} - {len(features)} items to {FEAT_OFP}")

            with open(FEAT_OFP, "wb") as f:
                pickle.dump(features, f)

    tokenizer.save_vocabulary(FEAT_DIR)
    print(f"Saved items and vocabulary to {FEAT_DIR}")
