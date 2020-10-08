import pandas as pd
import numpy as np
import argparse
# import tensorflow_hub as hub
# from sentence_transformers import SentenceTransformer


def add_use(basil):
    """
    Adds USE embeds from arXiv:1803.11175
    :param basil: DataFrame with BASIL instances and - selected - annotations
    :return: DataFrame with USE embeddings as a field
    """

    print('''
    
        Install tensorflow-hub by running:
        
        pip install "tensorflow>=2.0.0"
        pip install --upgrade tensorflow-hub
        
        And uncomment:
        # import tensorflow_hub as hub
        
        And:
        
        # embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        # em = embed([sent])
    
        ''')

    '''
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embs = []
    for sent in basil.sentence.values:
        em = embed([sent])
        em = list(np.array(em[0]))
        embs.append(em)
    basil['USE'] = embs
    basil.to_csv('data/w_embed/basil_w_USE.csv')
    '''


def add_sbert(basil):
    """
    Adds Sentence-BERT embeddings from arXiv:1908.10084
    :param basil: DataFrame with BASIL instances and - selected - annotations
    :return: DataFrame with Sentence-BERT embeddings as a field
    """

    print('''

        Install sentence-transformers by running:

        pip install sentence-transformers

        And uncomment:
        # from sentence_transformers import SentenceTransformer

        and this function
        ''')

    '''
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embs = model.encode(basil.sentence.values)
    embs = [list(el) for el in embs]
    basil['sbert'] = embs
    basil.to_csv('data/w_embed/basil_w_sbert.csv')
    '''


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


def preprocess_for_cim(basil, add_use=False, add_sbert=False, ofp="data/inputs/cim/cim_basil.tsv"):
    """
    Groups basil instances by story and source, and write .tsv lines of
    [id + article sent ids + event sent ids + label + index in article]
    :param
    basil: DataFrame with BASIL instances and - selected - annotations
    ofp: output file path of all instances
    :return: None, writes to ofp
    """

    # optionally, add embeddings for experimentation with different CIM base embeddings
    if add_use:
        add_use(basil)
    if add_sbert:
        add_sbert(basil)

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


def preprocess_for_tapt(basil, train_ofp = "data/tapt/basil_train.tsv", test_ofp = "data/tapt/basil_test.tsv"):
    """
    Split for tapt
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
    parser.add_argument('-add_use', '--add_use', action='store_true', default=False,
                        help='Add USE to BASIL voor CIM with USE input')
    parser.add_argument('-add_sbert', '--add_sbert', action='store_true', default=False,
                        help='Add Sentence-BERT to BASIL voor CIM with SBERT input')
    args = parser.parse_args()

    basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')

    # ARTICLE & CONTEXT
    # Groups basil instances by story and source, and write .tsv lines
    preprocess_for_cim(basil, add_use=False, add_sbert=False, ofp="data/inputs/cim/cim_basil.tsv")

    # DOMAIN CONTEXT
    # Split for tapt
    preprocess_for_tapt(basil, train_ofp="data/tapt/basil_train.tsv", test_ofp="data/tapt/basil_test.tsv")
