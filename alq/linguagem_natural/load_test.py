import bs4 as bs
import urllib.request
import nltk
import spacy
from spacy.matcher import PhraseMatcher
from IPython.core.display import HTML

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

pln = spacy.load("pt_core_news_sm")
string = "turing"
token_pesquisa = pln(string)
print(conteudo)
matcher = PhraseMatcher(pln.vocab)
matcher.add("SEARCH", None, token_pesquisa)
doc = pln(conteudo)
matches = matcher(doc)
print(matches)
numero_plavras = 50
matches = matcher(doc)
with open("data.html", "w") as file:
    file.write(f"<h1>{string.upper()}</h1>")
    file.write(f"""<p><strong>Resultados encontrados:</strong>{len(matches)}</p>""")
    texto = ""
    for i in matches:
        inicio = i[1] - numero_plavras
        if inicio < 0:
            inicio = 0
        texto += str(doc[inicio : i[2] + numero_plavras]).replace(
            string, f"<mark>{string}</mark>"
        )
        texto += "<br/> <br/>"
    file.write(f"""... {texto} ...""")
