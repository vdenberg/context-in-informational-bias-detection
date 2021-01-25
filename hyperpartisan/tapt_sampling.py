import xml.etree.ElementTree as ET

in_fp = '../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/articles-training-bypublisher-20181122.xml'

with open(in_fp, 'r') as f:
    content = f.readlines()
    content = ''.join(content[2:-1])
    tags_n_articles = content.split('</article>')

for tna in tags_n_articles[:10]:
    a = ''.join(tna.split('>')[2:])
    print(a[:10], a[-10:])
    print('---')

#for tna in tags_n_articles:
#    a = ''.join(tna.split('>')[2:])

