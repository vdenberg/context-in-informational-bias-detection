from __future__ import absolute_import, division, print_function
from transformers import RobertaTokenizer, BertTokenizer
import pickle
from lib.handle_data.PreprocessForPLM import BinaryClassificationProcessor
from lib.handle_data.PreprocessForPLM import convert_example_to_bert_feature, convert_example_to_roberta_feature
import os, time
from lib.handle_data.SplitData import split_input_for_plm
import argparse
import spacy


def preprocess(rows, model):
    count = 0
    total = len(rows)
    features = []
    for row in rows:
        if model == 'BERT':
            feats = convert_example_to_bert_feature(row)
        elif model == 'RoBERTa':
            feats = convert_example_to_roberta_feature(row)
        features.append(feats)
        count += 1

        if count % 250 == 0:
            status = f'Processed {count}/{total} rows'
            print(status)
    return features


parser = argparse.ArgumentParser()
parser.add_argument('-task', '--clf_task', help='sent_clf|tok_clf', type=str, default='sent_clf')  # 2,3,4
parser.add_argument('-plm', '--pretrained_lm', help='BERT|RoBERTa', type=str, default='RoBERTa')  # 5e-5, 3e-5, 2e-5
args = parser.parse_args()

CLF_TASK = args.clf_task
PLM = args.pretrained_lm

######
# structure of project
######

DATA_DIR = f'data/{CLF_TASK}_input/plm_input'
FEAT_DIR = f'data/{CLF_TASK}_input/features_for_{PLM}/'
DATA_TSV_IFP = os.path.join(DATA_DIR, f"plm_basil.tsv")
FEAT_OFP = os.path.join(FEAT_DIR, f"all_features.pkl")

######
# load data
# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
######

folds = split_input_for_plm(DATA_DIR, recreate=True, n_voters=1)
MAX_SEQ_LENGTH = 124
NR_FOLDS = len(folds)

######
# get relevant classes depending on task and model
######

dataloader = BinaryClassificationProcessor()
label_list = dataloader.get_labels(output_mode=CLF_TASK)  # [0, 1] for binary classification

if CLF_TASK == 'tok_clf':
    spacy_tokenizer = spacy.load("en_core_web_sm")
    if PLM == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)
    elif PLM == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False, do_basic_tokenize=False)
    label_map = {label: i + 1 for i, label in enumerate(label_list)}

elif CLF_TASK == 'sent_clf':
    spacy_tokenizer = None
    if PLM == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    elif PLM == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    label_map = {label: i for i, label in enumerate(label_list)}

######
# write features for whole dataset
# takes a while!
######

FORCE = False
if not os.path.exists(FEAT_OFP) or FORCE:
    examples = dataloader.get_examples(DATA_TSV_IFP, 'train', sep='\t')
    examples = [(ex, label_map, MAX_SEQ_LENGTH, tokenizer, spacy_tokenizer, CLF_TASK) for ex in examples if ex.text_a]

    features = preprocess(examples, model=PLM)
    features_dict = {feat.my_id: feat for feat in features}

    with open(FEAT_OFP, "wb") as f:
        pickle.dump(features, f)
    time.sleep(15)
else:
    with open(FEAT_OFP, "rb") as f:
       features = pickle.load(f)
       features_dict = {feat.my_id: feat for feat in features}

print(f"Processed fold all - {len(features)} items")

######
# write features in seperate files per fold
######

for fold in folds:
    fold_name = fold['name']
    for set_type in ['train', 'dev', 'test']:
        infp = os.path.join(DATA_DIR, f"{fold_name}_{set_type}.tsv")
        FEAT_OFP = os.path.join(FEAT_DIR, f"{fold_name}_{set_type}_features.pkl")

        examples = dataloader.get_examples(infp, set_type, sep='\t')
        features = [features_dict[example.my_id] for example in examples if example.text_a]
        print(f"Processed fold {fold_name} {set_type} - {len(features)} items and writing to {FEAT_OFP}")

        with open(FEAT_OFP, "wb") as f:
            pickle.dump(features, f)

tokenizer.save_vocabulary(FEAT_DIR)
