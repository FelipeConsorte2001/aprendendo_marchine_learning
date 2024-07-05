from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import bs4 as bs
import urllib.request
from spacy.lang.pt.stop_words import STOP_WORDS
import spacy

pln = spacy.load("pt_core_news_sm")

dados = urllib.request.urlopen(
    "https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial"
)
dados = dados.read()
# print(dados)
dados_html = bs.BeautifulSoup(dados, "lxml")
paragrafos = dados_html.find_all("p")
conteudo = ""
for p in paragrafos:
    conteudo += p.text.lower()

doc = pln(conteudo)
lista_token = []
for token in doc:
    lista_token.append(token.text)
sem_stop = []
for palavra in lista_token:
    if pln.vocab[palavra].is_stop == False:
        sem_stop.append(palavra)
color_map = ListedColormap(["orange", "green", "red", "magenta"])
cloud = WordCloud(background_color="white", max_words=100, colormap=color_map)
cloud = cloud.generate(" ".join(sem_stop))
plt.figure(figsize=(15, 15))
plt.imshow(cloud)
plt.axis("off")
plt.show()
