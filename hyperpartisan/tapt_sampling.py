from bs4 import BeautifulSoup
import random

in_fp = '../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/articles-training-bypublisher-20181122.xml'
out_fp = '../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/input.txt'

with open(in_fp, 'r') as f:
    content = f.readlines()
    content = ''.join(content[2:-1])
    tags_n_articles = content.split('</article>')

random.shuffle(tags_n_articles)

count = 0
sentences = []

for tna in tags_n_articles[:5500]:
    lines = tna.strip('\n').split('\n')
    try:
        tag, firstline = lines[0].split('<p>')
    except ValueError:
        print(lines[0])
        exit(0)
    firstline = '<p>' + firstline
    for l in [firstline] + lines[1:]:
        text = l[3:-5]
        text = BeautifulSoup(text, "lxml").text
        if len(text) > 5 & count <= 5000:
            sentences.append(text)

with open(out_fp, 'w') as f:
    for t in sentences:
        f.write(t)
        f.write('\n')

# test
# cat ../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/input.txt | wc -l