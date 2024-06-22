import pandas as pd
from apyori import apriori
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA

base_cartao = pd.read_csv("./../../database/credit_card_clients.csv", header=1)
base_cartao["BILL_TOTAL"] = (
    base_cartao["BILL_AMT1"]
    + base_cartao["BILL_AMT2"]
    + base_cartao["BILL_AMT3"]
    + base_cartao["BILL_AMT4"]
    + base_cartao["BILL_AMT5"]
    + base_cartao["BILL_AMT6"]
)
x_cartao_mais = base_cartao.iloc[:, [1, 2, 3, 4, 5, 25]].values
scaler_cartao_mais = StandardScaler()
x_cartao_mais = scaler_cartao_mais.fit_transform(x_cartao_mais)
wscc = []
for i in range(1, 11):
    KMeans_cartao = KMeans(n_clusters=i, random_state=0)
    KMeans_cartao.fit(x_cartao_mais)
    wscc.append(KMeans_cartao.inertia_)

grafico = px.line(x=range(1, 11), y=wscc)
print(wscc)
grafico.show()
KMeans_cartao_mais = KMeans(n_clusters=4, random_state=0)
rotulos = KMeans_cartao_mais.fit_predict(x_cartao_mais)
print(rotulos)
pca = PCA(n_components=2)
x_cartao_mais_pca = pca.fit_transform(x_cartao_mais)
print(x_cartao_mais_pca)
grafico2 = px.scatter(
    x=x_cartao_mais_pca[:, 0], y=x_cartao_mais_pca[:, 1], color=rotulos
)
grafico2.show()
lista_clientes = np.column_stack((base_cartao, rotulos))
print(lista_clientes)
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]
print(lista_clientes)
