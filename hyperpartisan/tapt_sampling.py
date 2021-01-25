from bs4 import BeautifulSoup as bs
import lxml

with open("../experiments/dont-stop-pretraining/data/hyperpartisan/unlabeled/articles-training-bypublisher-20181122.xml", "r") as f:
    content = f.readlines()
    content = "".join(content)
    bs_content = bs(content, "lxml")

result = bs_content.find_all("article")
print(len(result))