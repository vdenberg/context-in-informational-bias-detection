from bs4 import BeautifulSoup as bs
import random
import spacy

nlp = spacy.load('en_core_web_sm')
in_fp = '../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/articles-training-bypublisher-20181122.xml'
out_fp = '../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/input_docs.txt'

with open(in_fp, 'r') as f:
    content = f.readlines()
    content = ''.join(content[2:-1])
    tags_n_articles = content.split('</article>')

random.shuffle(tags_n_articles)

count = 0
sents = []

while count <= 5000:
    for tna in tags_n_articles:
        text = bs(tna, "lxml").text
        if len(text) > 5:
            sp_text = nlp(text)
            for s in sp_text.sents:
                sents.append(s.text)
        count += 1

print(count)

with open(out_fp, 'w') as f:
    for t in sents:
        f.write(t)
        f.write('\n')

# test
# cat ../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/input.txt | wc -l