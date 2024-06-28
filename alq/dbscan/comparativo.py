from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import plotly.express as px

x_random, y_random = datasets.make_moons(n_samples=1500, noise=0.09)
grafico = px.scatter(x=x_random[:, 0], y=x_random[:, 1])
grafico.show()
kmeans = KMeans(n_clusters=2)
rotulos = kmeans.fit_predict(x_random)
grafico = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos)
# grafico.show()

hc = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
rotulos2 = hc.fit_predict(x_random)
grafico2 = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos2)
grafico2.show()

dbscan = DBSCAN(eps=0.1)
rotulos3 = dbscan.fit_predict(x_random)
grafico3 = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos3)
grafico3.show()
