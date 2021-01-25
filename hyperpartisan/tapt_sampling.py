import xml.etree.ElementTree as ET
import pandas as pd
from bs4 import BeautifulSoup
import random

in_fp = '../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/articles-training-bypublisher-20181122.xml'
out_fp = '../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/input.txt'

with open(in_fp, 'r') as f:
    content = f.readlines()
    content = ''.join(content[2:-1])
    tags_n_articles = content.split('</article>')

random.shuffle(tags_n_articles)

texts = []
for tna in tags_n_articles[:5000]:
    lines = tna.strip('\n').split('\n')
    try:
        tag, firstline = lines[0].split('<p>')
    except ValueError:
        print(lines[0])
    firstline = '<p>' + firstline
    text_lines = []
    for l in [firstline] + lines[1:]:
        text_lines.append(l[3:-5])
    text = ' '.join(text_lines)
    cleantext = BeautifulSoup(text, "lxml").text
    texts.append(cleantext)

with open(out_fp, 'w') as f:
    for t in texts:
        f.write(t)
        f.write('\n')

# test
# cat ../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/input.txt | wc -l