from sklearn.datasets import make_blobs
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

x_random, y_random = make_blobs(n_samples=200, centers=5, random_state=1)
print(x_random, y_random)
kmeans_blobs = KMeans(n_clusters=5)
kmeans_blobs.fit(x_random)
rotulos = kmeans_blobs.predict(x_random)
centroides = kmeans_blobs.cluster_centers_
grafico1 = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos)

grafico2 = px.scatter(x=centroides[:, 0], y=centroides[:, 1], size=[5, 5, 5, 5, 5])
grafico3 = go.Figure(data=grafico1.data + grafico2.data)
grafico3.show()
