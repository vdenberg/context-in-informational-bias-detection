import xml.etree.ElementTree as ET

in_fp = '../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/articles-training-bypublisher-20181122.xml'

tree = ET.parse(in_fp)
root = tree.getroot()

for member in root.findall('article'):
    print(member)
    print(member.find('article').text)