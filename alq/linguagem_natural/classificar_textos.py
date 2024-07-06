import spacy
import string
import random
import seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from spacy.lang.pt.stop_words import STOP_WORDS

base_dados = pd.read_csv("./../../database/base_treinamento.txt", encoding="utf-8")


print(base_dados.head())
sns.countplot(base_dados["emocao"], label="contagem")
# plt.show()

pontuacoes = string.punctuation
stop_words = STOP_WORDS
pln = spacy.load("pt_core_news_sm")


def preprocessamento(texto):

    documento = pln(texto.lower())
    lista = []
    for token in documento:
        # lista.append(token.text)
        lista.append(token.lemma_)
    lista = [
        palavra
        for palavra in lista
        if palavra not in stop_words and palavra not in pontuacoes
    ]
    lista = " ".join([str(elemento) for elemento in lista if not elemento.isdigit()])
    return lista


teste = preprocessamento(
    "Estou aprendendo processamento de linguagem natiral, curso em são paulo"
)

base_dados["texto"] = base_dados["texto"].apply(preprocessamento)
print(teste)

base_dados_final = []
for texto, emocao in zip(base_dados["texto"], base_dados["emocao"]):
    dic = {}
    if emocao == "alegria":
        dic = {"ALEGRIA": True, "MEDO": False}
    elif emocao == "medo":
        dic = {"ALEGRIA": False, "MEDO": True}
    base_dados_final.append([texto, dic.copy()])
print(len(base_dados_final))
print(base_dados_final[0])
modelo = spacy.blank("pt")
modelo.add_pipe("textcat")
categorias = modelo.get_pipe("textcat")

categorias.add_label("ALEGRIA")
categorias.add_label("MEDO")
historico = []
