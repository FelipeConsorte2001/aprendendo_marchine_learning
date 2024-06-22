import pandas as pd
from apyori import apriori
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

base_cartao = pd.read_csv("./../../database/credit_card_clients.csv", header=1)
base_cartao["BILL_TOTAL"] = (
    base_cartao["BILL_AMT1"]
    + base_cartao["BILL_AMT2"]
    + base_cartao["BILL_AMT3"]
    + base_cartao["BILL_AMT4"]
    + base_cartao["BILL_AMT5"]
    + base_cartao["BILL_AMT6"]
)
x_cartao = base_cartao.iloc[:, [1, 25]].values

print(base_cartao)
scaler_cartao = StandardScaler()
x_cartao = scaler_cartao.fit_transform(x_cartao)

wscc = []
for i in range(1, 11):
    KMeans_cartao = KMeans(n_clusters=i, random_state=0)
    KMeans_cartao.fit(x_cartao)
    wscc.append(KMeans_cartao.inertia_)
grafico = px.line(x=range(1, 11), y=wscc)
print(wscc)
grafico.show()
kmeans_cartao = KMeans(n_clusters=4, random_state=0)
rotulos = kmeans_cartao.fit_predict(x_cartao)
grafico = px.scatter(x=x_cartao[:, 0], y=x_cartao[:, 1], color=rotulos)
grafico.show()
lista_clientes = np.column_stack((base_cartao, rotulos))
print(lista_clientes)
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]
print(lista_clientes)
