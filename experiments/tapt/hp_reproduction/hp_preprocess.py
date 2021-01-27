from bs4 import BeautifulSoup as bs
import random, argparse, os, json
import spacy


def preprocess_labeled(input_fp, output_fp):
    with open(input_fp) as f:
        content = [json.loads(el) for el in f.readlines()]
        eval_docs = [el['text'] for el in content]

    nlp = spacy.load('en_core_web_sm')
    with open(output_fp, 'w') as f:
        for t in eval_docs:
            doc = nlp(t)
            for sentence in doc.sents:
                t = sentence.text
                t = t.strip()
                f.write(t)
                f.write('\n')

    print(f'{len(eval_docs)} item to {output_fp}')


def remove_lhml(tagged):
    parsed = bs(tagged, "lxml")
    text = parsed.text
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-curated', '--curated', action='store_true', default=False, help='prep xml docs?')
    parser.add_argument('-labeled', '--labeled', action='store_true', default=False, help='prep jsonl files?')
    parser.add_argument('-cursize', '--curated_size', type=int, default=5000)
    args = parser.parse_args()

    LABELED = args.labeled
    CURATED = args.curated
    SIZE = args.curated_size

    if LABELED:
        # fps
        train_fp = '../data/hyperpartisan/train.jsonl'
        train_out_fp = '../data/hyperpartisan/train.txt'

        eval_fp = '../data/hyperpartisan/dev.jsonl'
        eval_out_fp = '../data/hyperpartisan/eval.txt'

        preprocess_labeled(train_fp, train_out_fp)
        preprocess_labeled(eval_fp, eval_out_fp)

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

