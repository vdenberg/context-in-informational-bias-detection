from bs4 import BeautifulSoup as bs
import random, argparse, os, json, pickle, time
import spacy
import pandas as pd

from nltk.tokenize import sent_tokenize
from transformers import RobertaTokenizer, BertTokenizer
from lib.handle_data.PreprocessForPLM import BinaryClassificationProcessor, InputFeatures
from lib.handle_data.PreprocessForPLM import convert_basil_for_plm_inputs, convert_example_to_bert_feature, convert_example_to_roberta_feature


def load_labeled(in_dir):
    labeled = []
    for set_type in ['train', 'dev', 'test']:
        fp = os.path.join(in_dir, f'{set_type}.jsonl')
        with open(fp) as f:
            for el in f.readlines():
                instance = json.loads(el)
                instance['set_type'] = set_type
                labeled.append(instance)
    print(f'Loaded {len(labeled)} instances')
    return labeled


def sentence_split(docs):
    print(docs[:10])
    stats = {}
    output = []
    for el in docs:
        sents = sent_tokenize(el['text'])
        for s in sents:
            o = el.copy()
            o['text'] = s
            stats.setdefault(el['set_type'], 0)
            stats[el['set_type']] += 1
            output.extend(o)
    print(f'Turned {len(docs)} doc instances into {len(output)} sentence instances')
    print(stats)
    return output


def preprocess_for_plm(rows, model):
    """
    Loops over instances and chooses correct conversion to numeric PLM features
    :param rows: rows of instances with tokenizer and other objects needed for preprocessing
    :param model: string that specifies which model to preprocess for
    :return:
    """
    count = 0
    features = []
    for row in rows:
        if model == 'bert':
            feats = convert_example_to_bert_feature(row)
        elif model == 'roberta':
            feats = convert_example_to_roberta_feature(row)
        features.append(feats)
        count += 1
    return features


def preprocess_for_dsp_run_ml(docs, output_fp):
    nlp = spacy.load('en_core_web_sm')
    with open(output_fp, 'w') as f:
        for t in docs:
            doc = nlp(t)
            for sentence in doc.sents:
                t = sentence.text
                t = t.strip()
                f.write(t)
                f.write('\n')

    print(f'{len(docs)} item to {output_fp}')


def remove_lhml(tagged):
    parsed = bs(tagged, "lxml")
    text = parsed.text
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-labeled', '--dsp_labeled', action='store_true', default=False, help='prep jsonl files?')
    parser.add_argument('-cur', '--dsp_curated', action='store_true', default=False, help='prep xml docs?')
    parser.add_argument('-cursize', '--curated_size', type=int, default=5000)
    parser.add_argument('-plm', '--plm', type=str, default=None, help='bert|roberta')
    args = parser.parse_args()

    LABELED = args.dsp_labeled
    CURATED = args.dsp_curated
    SIZE = args.curated_size
    PLM = args.plm

    DATA_DIR = '../data/hyperpartisan/'

    if LABELED:
        # fps
        labeled = load_labeled(DATA_DIR)
        all_labeled = sentence_split(labeled)

        train_docs = [el['text'] for el in labeled if el['set_type'] == 'train']
        eval_docs = [el['text'] for el in labeled if el['set_type'] == 'dev']
        preprocess_for_dsp_run_ml(train_docs, os.path.join(DATA_DIR, 'train.txt'))
        preprocess_for_dsp_run_ml(eval_docs, os.path.join(DATA_DIR, 'eval.txt'))

        if PLM:
            all_tsv_ifp = os.path.join(DATA_DIR, 'all.tsv')
            all_feat_ofp = os.path.join(DATA_DIR, 'all_features.pkl')

            if not os.path.exists(all_tsv_ifp):
                print(all_labeled)
                df = pd.DataFrame(all_labeled)
                print(df.head())
                df['alpha'] = ['a'] * len(df)
                df['label'] = [0 if el == 'false' else 1 for el in df['label']]
                df[['id', 'label', 'alpha', 'sentence']].to_csv(all_tsv_ifp, sep='\t', index=False, header=False)

                for st in ['train', 'dev', 'test']:
                    tsv_fp = os.path.join(DATA_DIR, f'{st}.tsv')
                    df[df.set_type == st][['id', 'label', 'alpha', 'sentence']].to_csv(tsv_fp, sep='\t', index=False, header=False)

            dataloader = BinaryClassificationProcessor()
            label_list = dataloader.get_labels(output_mode='sent_clf')  # [0, 1] for binary classification

            if PLM == 'bert':
                tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
            elif PLM == 'roberta':
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            label_map = {label: i for i, label in enumerate(label_list)}
            MAX_SEQ_LEN = 124 #124

            if not os.path.exists(all_feat_ofp):
                examples = dataloader.get_examples(all_tsv_ifp, 'train', sep='\t')
                examples = [(ex, label_map, MAX_SEQ_LEN, tokenizer, None, 'sent_clf') for ex in examples if ex.text_a]
                features = preprocess_for_plm(examples, model=PLM)
                features_dict = {feat.my_id: feat for feat in features}

                with open(all_feat_ofp, "wb") as f:
                    pickle.dump(features, f)
                time.sleep(15)
            else:
                with open(all_feat_ofp, "rb") as f:
                    features = pickle.load(f)
                    features_dict = {feat.my_id: feat for feat in features}

            for set_type in ['train', 'dev', 'test']:
                infp = os.path.join(DATA_DIR, f"{set_type}.tsv")
                FEAT_OFP = os.path.join(DATA_DIR, f"{set_type}_features.pkl")

                if not os.path.exists(FEAT_OFP):
                    examples = dataloader.get_examples(infp, set_type, sep='\t')

                    features = [features_dict[example.my_id] for example in examples
                                if example.text_a and (example.my_id in features_dict)]

                    with open(FEAT_OFP, "wb") as f:
                        pickle.dump(features, f)

    if CURATED:
        # fps
        unlabeled_fp = '../data/hyperpartisan/unlabeled/articles-training-bypublisher-20181122.xml'
        if not os.path.exists(unlabeled_fp):
            print(f""" Please download unlabeled semeval hyperpartisan data from zenodo to {unlabeled_fp}""")
        unlab_out_fp = '../data/hyperpartisan/curated.txt'

        # read documents
        with open(unlabeled_fp, 'r') as f:
            content = f.readlines()
            content = ''.join(content[2:-1])
            tags_n_articles = content.split('</article>')

        # shuffle
        random.shuffle(tags_n_articles)

        # parse sample
        count = 0
        docs = []

        for tna in tags_n_articles:
            if count < SIZE:
                text = remove_lhml(tna)
                text = text.strip()
                #text = text.replace('\n', ' ')
                if len(text) > 5:
                    docs.append(text)
                    count += 1
            else:
                break

        # write
        with open(unlab_out_fp, 'w') as f:
            for t in docs:
                t = t.strip()
                f.write(t)
                f.write('\n')

        print(f'Size: {count}, sampled: {len(docs)}')

