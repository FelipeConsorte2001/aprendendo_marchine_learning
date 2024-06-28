import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import plotly.express as px

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
print(x_cartao)
dbscan_cartao = DBSCAN(eps=0.37, min_samples=5)
rotulos = dbscan_cartao.fit_predict(x_cartao)
# print(rotulos)
# print(np.unique(rotulos, return_counts=True))
grafico = px.scatter(x=x_cartao[:, 0], y=x_cartao[:, 1], color=rotulos)
grafico.show()
