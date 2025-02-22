import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

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
dendrograma = dendrogram(linkage(x_cartao, method="ward"))
# plt.show()
hc_cartao = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward")
rotulos = hc_cartao.fit_predict(x_cartao)
print(rotulos)
grafico = px.scatter(x=x_cartao[:, 0], y=x_cartao[:, 1], color=rotulos)
grafico.show()
