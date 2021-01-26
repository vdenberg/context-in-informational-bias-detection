from bs4 import BeautifulSoup as bs
import random, argparse, os, json
import spacy


def remove_lhml(tagged):
    parsed = bs(tagged, "lxml")
    text = parsed.text
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_docs', '--no_docs', action='store_true', default=False, help='skip prep docs?')
    parser.add_argument('-no_eval', '--no_eval', action='store_true', default=False, help='skip prep eval?')
    parser.add_argument('-size', '--size', type=int, default=5000)
    args = parser.parse_args()

    DOCS = not args.no_docs
    SIZE = args.size
    EVAL = not args.no_eval

    if DOCS:
        # fps
        unlabeled_fp = '../data/hyperpartisan/unlabeled/articles-training-bypublisher-20181122.xml'
        if not os.path.exists(unlabeled_fp):
            print(f""" Please download unlabeled semeval hyperpartisan data from zenodo to {unlabeled_fp}""")
        doc_fp = '../data/hyperpartisan/docs.txt'

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
        with open(doc_fp, 'w') as f:
            for t in docs:
                t = t.strip()
                f.write(t)
                f.write('\n')

        print(f'Size: {count}, sampled: {len(docs)}')

    if EVAL:
        # fps
        eval_fp = '../data/hyperpartisan/dev.jsonl'
        out_eval_fp = '../data/hyperpartisan/unlabeled/eval.txt'

        with open(eval_fp) as f:
            content = [json.loads(el) for el in f.readlines()]
            eval_docs = [el['text'] for el in content]

        nlp = spacy.load('en_core_web_sm')
        with open(out_eval_fp, 'w') as f:
            for t in eval_docs:
                doc = nlp(text)
                for sentence in doc.sents:
                    f.write(sentence.text)
                    f.write('\n')

        print(f'Eval size: {len(eval_docs)}')

    # test
    # cat ../experiments/dont-stop-pretraining/data/hp_reproduction/unlabeled/input.txt | wc -l

