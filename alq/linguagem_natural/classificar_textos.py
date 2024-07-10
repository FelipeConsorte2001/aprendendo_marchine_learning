import spacy
import string
import random
import seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from spacy.lang.pt.stop_words import STOP_WORDS
from spacy.training import Example

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
modelo.begin_training()
for epoca in range(1000):
    random.shuffle(base_dados_final)
    losses = {}
    for batch in spacy.util.minibatch(base_dados_final, 30):
        textos = [modelo(texto) for texto, entities in batch]
        annotations = [{"cats": entities} for texto, entities in batch]
        examples = [
            Example.from_dict(doc, annotation)
            for doc, annotation in zip(textos, annotations)
        ]
        modelo.update(examples, losses=losses)
    if epoca % 100 == 0:
        print(losses)
        historico.append(losses)
historico_loss = []
for i in historico:
    historico_loss.append(i.get("textcat"))
historico_loss = np.array(historico_loss)
print(historico_loss)
plt.plot(historico_loss)
plt.title("progressão do erro")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.show()
modelo.to_disk("modelo")
modelo_carregado = spacy.load("modelo")
texto_positivo = "eu adoro a cor dos seus olhos"
texto_positivo = preprocessamento(texto_positivo)
previsao = modelo_carregado(texto_positivo)
print(previsao.cats)
previsao = modelo_carregado(preprocessamento("estou com medo dele"))
print(previsao.cats)
