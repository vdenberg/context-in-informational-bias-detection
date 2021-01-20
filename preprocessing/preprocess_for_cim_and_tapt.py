import pandas as pd
import argparse, os
from lib.handle_data.BasilLoader import LoadBasil
from lib.handle_data.SplitData import Split
import spacy
import json, re


def tokenize(x):
    global nlp
    return [token.text for token in nlp(x)]


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
        full_string += [str(label), str(index)]

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

    if not os.path.exists(ofp):
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


def preprocess_basil_for_tapt(basil, test_size=50, train_ofp="data/tapt/basil_train.txt", test_ofp="data/tapt/basil_test.txt"):
    """
    Split for tapt
    """

    article_counter = 0

    for n, gr in basil.groupby('article'):

        if article_counter <= test_size:
            file_path = test_ofp
        else:
            file_path = train_ofp

        if file_path:
            with open(file_path, 'a') as f:
                sentences = gr.sentence.values
                for s in sentences:
                    f.write(s)
                    f.write('\n')

        article_counter += 1


def write_for_dsp(data, fp):
    with open(fp, 'w') as f:
        f.write('')

    with open(fp, 'a') as f:
        sentences = data.sentence.values
        labels = data.bias.values
        for s, l in zip(sentences, labels):
            instance = {'text': s, 'label': str(l), 'metadata': []}
            json.dump(instance, f)
            f.write('\n')


def preprocess_basil_for_dsp(data, data_dir, recreate=False):
    ''' This function selects those columns which are relevant for creating input for finetuning with DSP
    code our data, and saves them for each fold seperately. '''
    data['id'] = data.index

    # split data into folds
    spl = Split(data, which='both', recreate=False, sv=99)
    folds = spl.apply_split(features=['id', 'bias', 'sentence'])

    # write data for each fold
    for i, fold in enumerate(folds):
        fold_dir = os.path.join(data_dir, str(fold['name']))
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)

        test_ofp = os.path.join(fold_dir, f"test.jsonl")
        train_ofp = os.path.join(fold_dir, f"train.jsonl")
        dev_ofp = os.path.join(fold_dir, f"dev.jsonl")

        if not os.path.exists(train_ofp) or recreate:
            write_for_dsp(fold, train_ofp)

        if not os.path.exists(dev_ofp) or recreate:
            write_for_dsp(fold, dev_ofp)

        if not os.path.exists(test_ofp) or recreate:
            write_for_dsp(fold, test_ofp)

    return folds


def preprocess_cc_for_tapt(train_ifp="data/inputs/tapt/cc/fox", train_ofp="data/tapt/basil_train.txt"):
    """
    Preprocess commoncrawl data for tapt
    """

    files = os.listdir(train_ifp)
    files = [fn for fn in files if not fn == 'stat.json']

    for fn in files:
        ifp = os.path.join(train_ifp, fn)
        content = json.load(open(ifp))
        text = content['maintext']
        sentences = [s.text for s in nlp(text).sents]

        with open(train_ofp, 'a') as f:
            for s in sentences:
                f.write(s)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-add_use', '--add_use', action='store_true', default=False,
                        help='Add USE to BASIL voor CIM with USE input')
    parser.add_argument('-add_sbert', '--add_sbert', action='store_true', default=False,
                        help='Add Sentence-BERT to BASIL voor CIM with SBERT input')
    args = parser.parse_args()

    if not os.path.exists('data/inputs/sent_clf'):
        os.makedirs('data/inputs/sent_clf')
    if not os.path.exists('data/inputs/tok_clf'):
        os.makedirs('data/inputs/tok_clf')
    if not os.path.exists('data/inputs/seq_sent_clf'):
        os.makedirs('data/inputs/seq_sent_clf')
    if not os.path.exists('data/inputs/cim'):
        os.makedirs('data/inputs/cim')

    basil = LoadBasil().load_basil_raw()
    basil.to_csv('data/basil.csv')
    basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')

    # tokenize
    #nlp = spacy.load("en_core_web_sm")
    #basil['tokens'] = basil.sentence.apply(tokenize)
    #basil.to_csv('data/inputs/basil_w_tokens.csv')

    # ARTICLE & CONTEXT
    # Groups basil instances by story and source, and write .tsv lines
    #preprocess_for_cim(basil, add_use=False, add_sbert=False, ofp="data/inputs/cim/cim_basil.tsv")

    # DOMAIN CONTEXT
    # Split for tapt
    #preprocess_basil_for_tapt(basil, test_size=250, train_ofp="data/inputs/tapt/basil_train.txt", test_ofp="data/inputs/tapt/basil_test.txt")
    preprocess_basil_for_dsp(basil, data_dir="experiments/dont-stop-pretraining/basil_data/")

    # Split for source-specific tapt
    for source in ['fox', 'nyt', 'hpo']:
        #preprocess_basil_for_tapt(basil[basil['source'] == source], test_size=int(250 / 3), train_ofp="", test_ofp="data/inputs/tapt/basil_fox_test.txt")
        #preprocess_cc_for_tapt(train_ifp=os.path.join("data/inputs/tapt/cc/", source), train_ofp=os.path.join("data/inputs/tapt/", source + '_train.txt'))
        preprocess_basil_for_dsp(basil[basil['source'] == source],
                                 data_dir=f"experiments/dont-stop-pretraining/basil_data/{source}",
                                 recreate=True)

