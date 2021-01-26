from bs4 import BeautifulSoup as bs
import random, argparse, os, json


def remove_lhml(tagged):
    parsed = bs(tagged, "lxml")
    text = parsed.text
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-docs', '--docs', action='store_true', default=True, help='prep docs?')
    parser.add_argument('-eval', '--eval', action='store_true', default=True, help='prep eval?')
    parser.add_argument('-size', '--size', type=int, default=5000)
    args = parser.parse_args()

    DOCS = args.docs
    SIZE = args.size
    EVAL = args.eval

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
        while count <= SIZE:
            for tna in tags_n_articles:
                text = remove_lhml(tna)
                if len(text) > 5:
                    docs.append(text)
                count += 1

        # write
        with open(doc_fp, 'w') as f:
            for t in docs:
                f.write(t)
                f.write('\n')

        print(f'Size: {count}, sampled: {len(docs)}')

    if EVAL:
        # fps
        eval_fp = '../data/hyperpartisan/dev.jsonl'
        out_eval_fp = '../data/hyperpartisan/unlabeled/eval.txt'

        with open(eval_fp) as f:
            content = json.load(f)
            eval_docs = [el['text'] for el in content]

        with open(out_eval_fp, 'w') as f:
            for t in eval_docs:
                f.write(t)
                f.write('\n')

        print(f'Eval size: {len(text)}')





    # test
    # cat ../experiments/dont-stop-pretraining/data/hp_reproduction/unlabeled/input.txt | wc -l

