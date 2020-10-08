import argparse
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from lib.handle_data.BasilLoader import LoadBasil


def convert_for_plm(basil, task='sent_clf', ofp='data/tok_clf/plm_basil.csv'):
    """
    Select relevant columns for input to huggingface implementations of Pre-trained Language Models
    BERT and RoBERTa for Sentence and Token classification.
    :param
    basil: original BASIL DataFrame
    task: token or sentence classification
    ofp: output file path of all instances
    :return: None, writes to ofp
    """
    basil['id'] = basil['uniq_idx.1'].str.lower()
    basil['alpha'] = ['a'] * len(basil)

    if task == 'sent_clf':
        basil = basil.rename(columns={'bias': 'label'})
    elif task == 'tok_clf':
        basil = basil.rename(columns={'inf_start_ends': 'label'})

    basil.to_csv(ofp, sep='\t', index=False, header=False)


def add_USE(basil):
    """
    Adds USE embeds from arXiv:1803.11175
    :param basil: DataFrame with BASIL instances and - selected - annotations
    :return: DataFrame with USE embeddings as a field
    """

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embs = []
    for sent in basil.sentence.values:
        em = embed([sent])
        em = list(np.array(em[0]))
        embs.append(em)
    basil['USE'] = embs
    basil.to_csv('data/w_embed/basil_w_USE.csv')


def add_sbert(basil):
    """
    Adds Sentence-BERT embeddings from arXiv:1908.10084
    :param basil: DataFrame with BASIL instances and - selected - annotations
    :return: DataFrame with Sentence-BERT embeddings as a field
    """

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embs = model.encode(basil.sentence.values)
    embs = [list(el) for el in embs]
    basil['sbert'] = embs
    basil.to_csv('data/w_embed/basil_w_sbert.csv')


def convert_to_cim_instance(group, art_sent_ids, all_ev_ids):
    """
    Makes single line for cim_basil.tsv file consisting that looks like this:
    target sentence id \t article sentence ids \t event sentence ids \t label \t index
    :param group: DataFrame with bias labels of target story
    :param art_sent_ids: sentence ids of target article
    :param all_ev_ids: sentence ids of other articles on same event
    :return:
    """

    # convert ids to string to string
    art_string = ' '.join(art_sent_ids)
    ev_strings = []
    for ev_ids in all_ev_ids:
        ev_strings.append(' '.join(ev_ids))

    instances = []
    for index in range(len(group)):
        # get target sentence id
        uniq_id = art_sent_ids[index].lower()

        # get target label
        label = str(group.bias.values[index])

        # complete string
        full_string = [uniq_id, art_string]
        for ev_str in ev_strings:
            full_string.append(ev_str)
        full_string.append([label, index])

        instance = '\t'.join(full_string)
        instances.append(instance)

    return instances


def write_cim_input(basil, ofp="data/sent_clf/cim_basil.tsv"):
    """
    Groups basil instances by story and source, and write .tsv lines of
    [id + article sent ids + event sent ids + label + index in article]
    :param
    basil: DataFrame with BASIL instances and - selected - annotations
    ofp: output file path of all instances
    :return: None, writes to ofp
    """

    with open(ofp, 'w') as f:

        for n, ev_gr in basil.groupby(['story']):
            for src, art_gr in ev_gr.groupby('source'):

                # collect unique ids of instances (target sentences)
                art_ids = ev_gr[ev_gr.source == src]['uniq_idx.1'].to_list()

                if src == 'hpo':
                    ev1_ids = ev_gr[ev_gr.source == 'nyt']['uniq_idx.1'].to_list()
                    ev2_ids = ev_gr[ev_gr.source == 'fox']['uniq_idx.1'].to_list()

                elif src == 'nyt':
                    ev1_ids = ev_gr[ev_gr.source == 'hpo']['uniq_idx.1'].to_list()
                    ev2_ids = ev_gr[ev_gr.source == 'fox']['uniq_idx.1'].to_list()
                elif src == 'fox':
                    ev1_ids = ev_gr[ev_gr.source == 'hpo']['uniq_idx.1'].to_list()
                    ev2_ids = ev_gr[ev_gr.source == 'nyt']['uniq_idx.1'].to_list()

                group_lines = convert_to_cim_instance(art_gr, art_ids, [ev1_ids, ev2_ids])

                for line in group_lines:
                    f.write(line)
                    f.write('\n')


class WriteInput:
    """
    This is where inputs folder is filled with preprocessed data
    """

    def __init__(self):
        self.raw_basil = LoadBasil().load_basil_raw()
        self.write_basil_csv()
        self.basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')

    def write_basil_csv(self):
        self.basil.to_csv('data/basil.csv')

    def write_baseline_inputs(self):
        """
        # 1) Baselines:
        # Select relevant columns for input to huggingface implementations of Pre-trained Language Models
        # BERT and RoBERTa for Sentence and Token classification.
        """
        convert_for_plm(self.basil, 'sent_clf', ofp='data/inputs/sent_clf/plm_input/plm_basil.tsv')
        convert_for_plm(self.basil, 'tok_clf', ofp='data/inputs/tok_clf/plm_input/plm_basil.tsv')

    def write_ssc_inputs(self):
        """
        # 2) Direct Textual Context:
        # Uses plm input
        """
        raise NotImplementedError

    def write_cim_inputs(self, add_USE=False, add_sbert=False):
        """
        # 3) Writes input for Article Context & Event Context
        # - Optionally adds USE (arXiv:1803.11175) and Sentence-BERT embeddings (arXiv:1908.10084)
        # - Create CIM input that has target sentence and context information on the same line.
        """
        # optionally, add embeddings for experimentation with different CIM base embeddings
        if add_USE:
            add_USE(self.basil)
        if add_sbert:
            add_sbert(self.basil)

        write_cim_input(basil, ofp="data/inputs/cim/cim_basil.tsv")

    def write_domain_inputs(self):
        # 4) Domain Context
        # 3.1.) Split train & test input for Task-Adaptation of Roberta on BASIL.
        write_tapt_input(self.basil, train_ofp="data/inputs/tapt/tapt_basil_train.tsv", test_ofp="data/inputs/tapt/tapt_basil_test.tsv")
