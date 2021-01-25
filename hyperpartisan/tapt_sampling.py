import xml.etree.ElementTree as ET

in_fp = '../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/articles-training-bypublisher-20181122.xml'

with open(in_fp, 'r') as f:
    content = f.readlines()
    content = ''.join(content[2:-1])
    tags_n_articles = content.split('</article>')

for tna in tags_n_articles[:2]:
    lines = tna.split('\n')
    print(lines[0])
    tag, firstline = lines[0].split('<p>')
    firstline = '<p>' + firstline
    text_lines = []
    for l in [firstline] + lines[1:]:
        text_lines.append(l[3:-5])
    print(tag)
    print(text_lines[:3])
    print('---')


#for tna in tags_n_articles:
#    a = ''.join(tna.split('>')[2:])

