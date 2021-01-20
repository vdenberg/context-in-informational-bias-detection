from pprint import pprint

import os
import json
import pandas as pd
import numpy as np
import json
from transformers import BertTokenizer
import re
import spacy


def standardise_id(basil_id):
    if not basil_id[1].isdigit():
        basil_id = '0' + basil_id
    if not basil_id[-2].isdigit():
        basil_id = basil_id[:-1] + '0' + basil_id[-1]
    return basil_id.lower()


def load_basil_spans(start_ends):
    start_ends = start_ends[2:-2]
    if list(start_ends):
        start_ends = re.sub('\), \(', ';', start_ends)
        start_ends = start_ends.split(';')
        start_ends = [tuple(map(int, s_e.split(', '))) for s_e in start_ends]
    return start_ends


class LoadBasil:
    """
    This is where the content of basil.csv is determined
    """
    def __init__(self):
        self.raw_dir = 'data/emnlp19-BASIL/data/'

    def load_basil_all(self):
        # load exactly as published by authors

        collection = {}
        for file in os.listdir(self.raw_dir):
            idx = int(file.split('_')[0])
            source = file.split('_')[1][:3]
            story = collection.setdefault(idx, {'entities': set(), 'hpo': None, 'fox': None, 'nyt': None})
            with open(self.raw_dir + file) as f:
                content = json.load(f)
            story[source] = content
            story['entities'].update(content['article-level-annotations']['author-sentiment'])

        return collection

    def load_basil_raw(self):
        # load raw but stripped off fields that are not as relevant
        pre_df = []
        for file in os.listdir(self.raw_dir):
            story = file.split('_')[0]
            source = file.split('_')[1][:3]
            with open(self.raw_dir + file) as f:
                file_content = json.load(f)

                # pprint(file_content)

                main_entities = file_content['main-entities']
                stance = file_content['article-level-annotations']['stance']

                sentences = file_content['body']
                for sent in sentences:
                    sentence = sent['sentence']
                    sent_idx = str(sent['sentence-index'])
                    lexical_ann = [ann for ann in sent['annotations'] if ann['bias'] == 'Lexical']
                    informational_ann = [ann for ann in sent['annotations'] if ann['bias'] == 'Informational']
                    lex_bias_present = 1 if lexical_ann else 0
                    inf_bias_present = 1 if informational_ann else 0

                    inf_start_ends = []
                    lex_start_ends = []

                    inf_targets = []
                    lex_targets = []

                    inf_polarities = []
                    lex_polarities = []

                    inf_directs = []
                    lex_directs = []

                    inf_quote = []

                    if inf_bias_present:
                        for ann in informational_ann:
                            inf_start_ends.append((ann['start'],ann['end']))
                            inf_targets.append(ann['target'])
                            inf_polarities.append(ann['polarity'])
                            inf_directs.append(ann['aim'])
                            inf_quote.append(ann['quote'])

                    if lex_bias_present:
                        for ann in lexical_ann:
                            lex_start_ends.append((ann['start'], ann['end']))
                            lex_targets.append(ann['target'])
                            lex_polarities.append(ann['polarity'])
                            lex_directs.append(ann['aim'])

                    pre_df.append([story, source, main_entities, sent_idx, lex_bias_present, inf_bias_present, sentence,
                                   inf_targets, lex_targets, inf_polarities, lex_polarities, inf_directs, lex_directs,
                                   lex_start_ends, inf_start_ends, stance, inf_quote])

        columns = ['story', 'source', 'main_entities', 'sent_idx', 'lex_bias', 'bias', 'sentence',
                   'inf_entities', 'lex_entities',  'inf_pol', 'lex_pol', 'inf_dir', 'lex_dir',
                   'lex_start_ends', 'inf_start_ends', 'stance', 'inf_quote']
        df = pd.DataFrame(pre_df, columns=columns)
        df['article'] = df.story.astype(str) + df.source
        df['uniq_idx'] = df['story'] + df['source'] + df['sent_idx']
        df['uniq_idx'] = df['uniq_idx'].apply(standardise_id)
        df = df.set_index(df['uniq_idx'])
        empty_sentences = ['46fox24', '48fox19', '11fox23', '47nyt19', '47fox22', '58fox62', '52fox18']
        df['label'] = df.bias
        df = df.drop(empty_sentences)
        return df


