import json
import os

import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize

from lib.handle_data.SplitData import Split


def write_for_dsp_lm(df, fp):
    sentences = list(df.sentence.values)
    with open(fp, 'w') as f:
        for s in sentences:
            assert isinstance(s, str)

            f.write(s)
            f.write('\n')


def preprocess_basil_for_lm(basil_df, eval_size=20, data_dir="data/tapt/basil_and_source_tapt/data", source=None):
    """
    Preprocess for language modeling
    """
    if source:
        train_ofp = os.path.join(data_dir, f'{source}_basil_train.txt')
        eval_ofp = os.path.join(data_dir, f'{source}_basil_eval.txt')
        eval_size = int(eval_size / 3)
    else:
        train_ofp = os.path.join(data_dir, 'basil_train.txt')
        eval_ofp = os.path.join(data_dir, 'basil_eval.txt')

    articles = basil_df.article.unique()
    basil_df = basil_df.set_index(keys='article', drop=False)
    train_df = basil_df.loc[articles[:eval_size]]
    eval_df = basil_df.loc[articles[eval_size:]]

    write_for_dsp_lm(train_df, train_ofp)
    write_for_dsp_lm(eval_df, eval_ofp)

    print("Basil for TAPT (run_language_modeling):")
    print(f'Wrote {len(train_df)} sentences to {train_ofp}')
    print(f'Wrote {len(eval_df)} sentences to {eval_ofp}\n')


def list_cc_files(cc_dir):
    cc_fns = os.listdir(cc_dir)
    cc_text_fns = [os.path.join(cc_dir, fn) for fn in cc_fns if not fn == 'stat.json']
    return cc_text_fns


def preprocess_cc_for_lm(cc_dir, tapt_dir, source):
    """
    Preprocess commoncrawl data for curated tapt
    """
    sources = (source,) if source else ('fox', 'nyt', 'hpo')

    cc_fns = []
    for s in sources:
        cc_s_dir = os.path.join(cc_dir, s)
        cc_fns.extend(list_cc_files(cc_s_dir))

    source_string = '_'.join(sources)
    ofp = os.path.join(tapt_dir, f'{source_string}_cur_train.txt')

    #if not os.path.exists(ofp):
    with open(ofp, 'w') as f:
        for fn in cc_fns:
            content = json.load(open(fn))
            text = content['maintext']
            sents = sent_tokenize(text) # nlp(text).sents
            sentences = [s.text.strip() + '\n' for s in sents]  # todo speed up
            for s in sentences:
                f.write(s)

    print("CC for Cur TAPT (run_language_modeling):")
    print(f'Wrote {len(cc_fns)} files to {ofp}\n')


def write_for_dsp_train(data, fp):
    with open(fp, 'w') as f:
        ids = data.id.values
        sentences = data.sentence.values
        labels = data.label.values
        for i, s, l in zip(ids, sentences, labels):
            instance = {'text': s, 'label': str(l), 'metadata': [i]}
            json.dump(instance, f)
            f.write('\n')


def preprocess_basil_for_dsp_train(data, data_dir, recreate=False, source=None):
    """ This function selects those columns which are relevant for creating input for fine-tuning with DSP
    code our data, and saves them for each fold separately. """

    data['id'] = data.index
    data_dir = os.path.join(data_dir, source)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # split data into folds
    spl = Split(data, which='both', recreate=False, sv=99)
    folds = spl.apply_split(features=['id', 'label', 'sentence'], source=source)

    data_str_fp = os.path.join(data_dir, 'stats.json')
    data_strs = {}

    # write data for each fold
    for i, fold in enumerate(folds):
        fold_dir = os.path.join(data_dir, str(fold['name']))
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)

        train_ofp = os.path.join(fold_dir, f"train.jsonl")
        dev_ofp = os.path.join(fold_dir, f"dev.jsonl")
        test_ofp = os.path.join(fold_dir, f"test.jsonl")

        if not os.path.exists(train_ofp) or recreate:
            write_for_dsp_train(fold['train'], train_ofp)

        if not os.path.exists(dev_ofp) or recreate:
            write_for_dsp_train(fold['dev'], dev_ofp)

        if not os.path.exists(test_ofp) or recreate:
            write_for_dsp_train(fold['test'], test_ofp)

        if source:
            name = f"{source}_basil_{fold['name']}"
        else:
            name = f"basil_{fold['name']}"

        stats_dir = '/'.join(fold_dir.split('/')[-2:])
        size = sum(fold['sizes'])
        tmp = {"data_dir": stats_dir + "/", "dataset_size": size}
        data_strs[name] = tmp

    with open(data_str_fp, 'a') as f:
        json.dump(data_strs, f)

    print("Basil for fine-tuning/eval (train):")
    print(f'Wrote {len(folds)} folds to {data_dir}')
    print(f'Wrote {len(data_strs.keys())} folds stats to {data_str_fp}\n')

    return folds


if __name__ == '__main__':

    TAPT_DATA_DIR = "experiments/tapt/basil_and_source_tapt/data"
    if not os.path.exists(TAPT_DATA_DIR):
        os.mkdir(TAPT_DATA_DIR)

    basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')
    nlp = spacy.load("en_core_web_sm")

    # TAPT
    for src in ['', 'fox', 'nyt', 'hpo']:
        if src:
            basil = basil[basil['source'] == src]

        preprocess_basil_for_lm(basil, eval_size=20, data_dir=TAPT_DATA_DIR, source=src)  # basic TAPT
        preprocess_cc_for_lm(cc_dir='data/inputs/tapt/cc', tapt_dir=TAPT_DATA_DIR, source=src)  # curated TAPT
        preprocess_basil_for_dsp_train(basil, data_dir=TAPT_DATA_DIR, recreate=True, source=src)  # for eval
