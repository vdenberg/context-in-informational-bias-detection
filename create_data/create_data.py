import argparse
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from lib.handle_data.LoadData import LoadBasil


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


def write_tapt_input(basil, train_ofp="data/tapt/basil_train.tsv", test_ofp="data/tapt/basil_test.tsv"):
    """
    Groups basil instances by story and source, and write .tsv lines of
    [id + article sent ids + event sent ids + label + index in article]
    :param
    basil: DataFrame with BASIL instances and - selected - annotations
    train_ofp: output file path of train instances
    test_ofp: output file path of test instances
    :return: None, writes to train_ofp and test_ofp
    """

    test_size = 250

    article_counter = 0
    for n, gr in basil.groupby('article'):

        if article_counter <= test_size:
            file_path = train_ofp
        else:
            file_path = test_ofp

        with open(file_path, 'a') as f:
            sentences = gr.sentence.values
            for s in sentences:
                f.write(s)
                f.write(' ')
            f.write('\n')
        article_counter += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-SSC', '--n_epochs', type=int, default=10)  # 2,3,4
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)  # 5e-5, 3e-5, 2e-5
    parser.add_argument('-bs', '--batch_size', type=int, default=16)  # 16, 21
    parser.add_argument('-load', '--load_from_ep', type=int, default=0)
    args = parser.parse_args()

    basil = LoadBasil().load_basil_raw()
    basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')

    # 1) Baselines (Sent clf and Tok clf)
    convert_for_plm(basil, 'sent_clf', ofp='data/inputs/sent_clf/plm_input/plm_basil.tsv')
    convert_for_plm(basil, 'tok_clf', ofp='data/inputs/tok_clf/plm_input/plm_basil.tsv')

    # 2) Direct Textual Context:
    # Uses plm input

    # 3) Article Context & Event Context
    # 2.1.) Optional: Add USE (arXiv:1803.11175) and Sentence-BERT embeddings (arXiv:1908.10084)
    # add_USE(basil)
    # add_sbert(basil)
    # 2.2.) Create CIM input that has target sentence and context information on the same line.
    write_cim_input(basil, ofp="data/inputs/cim/cim_basil.tsv")

    # 4) Domain Context
    # 3.1.) Split train & test input for Task-Adaptation of Roberta on BASIL.
    write_tapt_input(basil, train_ofp="data/inputs/tapt/tapt_basil_train.tsv", test_ofp="data/inputs/tapt/tapt_basil_test.tsv")
