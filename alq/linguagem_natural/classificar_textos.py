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
from sklearn.metrics import confusion_matrix, accuracy_score

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
print("*********")
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

previsoes = []
for texto in base_dados["texto"]:
    previsao = modelo_carregado(texto)
    previsoes.append(previsao.cats)
print(previsoes)
previsao_final = []
for previsao in previsoes:
    if previsao["ALEGRIA"] > previsao["MEDO"]:
        previsao_final.append("alegria")
    else:
        previsao_final.append("medo")

previsao_final = np.array(previsao_final)
print(previsao_final)
respostas_reais = base_dados["emocao"].values
print(accuracy_score(respostas_reais, previsao_final))
cm = confusion_matrix(respostas_reais, previsao_final)
print(cm)
# avaliação base de dados
base_dados_teste = pd.read_csv("./../../database/base_teste.txt", encoding="utf-8")

base_dados_teste["texto"] = base_dados_teste["texto"].apply(preprocessamento)
previsoes = []
for texto in base_dados_teste["texto"]:
    previsao = modelo_carregado(texto)
    previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
    if previsao["ALEGRIA"] > previsao["MEDO"]:
        previsoes_final.append("alegria")
    else:
        previsoes_final.append("medo")

previsoes_final = np.array(previsoes_final)
respostas_reais = base_dados_teste["emocao"].values
print(accuracy_score(respostas_reais, previsoes_final))
cm2 = confusion_matrix(respostas_reais, previsoes_final)
print(cm2)
