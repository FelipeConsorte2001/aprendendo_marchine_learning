import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import ResidualsPlot
from sklearn.metrics import mean_absolute_error, mean_squared_error

base_casas = pd.read_csv("./../../database/house_prices.csv")
base_casas.drop("date", axis="columns", inplace=True)
print(base_casas)
print(base_casas.describe)
print(base_casas.isnull().sum())
print(base_casas.corr())
figura = plt.figure(figsize=(20, 20))
sns.heatmap(base_casas.corr(), annot=True)
plt.show()
x_casas = base_casas.iloc[:, 4:5].values
print(x_casas)
y_casas = base_casas.iloc[:, 1].values
print(y_casas)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = (
    train_test_split(x_casas, y_casas, test_size=0.3, random_state=0)
)
print(x_casas_treinamento.shape, y_casas_treinamento.shape)

regressor_simples_casas = LinearRegression()
regressor_simples_casas.fit(x_casas_treinamento, y_casas_treinamento)
print(regressor_simples_casas.intercept_)
print(regressor_simples_casas.coef_)
print(regressor_simples_casas.score(x_casas_treinamento, y_casas_treinamento))
print(regressor_simples_casas.score(x_casas_teste, y_casas_teste))
previsoes = regressor_simples_casas.predict(x_casas_treinamento)
print(previsoes)
grafico = px.scatter(x=x_casas_treinamento.ravel(), y=previsoes)
grafico.show()
grafico1 = px.scatter(x=x_casas_treinamento.ravel(), y=y_casas_treinamento)
grafico2 = px.line(x=x_casas_treinamento.ravel(), y=previsoes)
grafico2.data[0].line.color = "red"
grafico3 = go.Figure(data=grafico1.data + grafico2.data)
grafico3.show()
previsoes_teste = regressor_simples_casas.predict(x_casas_teste)
print(previsoes_teste)
print(abs(y_casas_teste - previsoes_teste).mean())
print(mean_absolute_error(y_casas_teste, previsoes_teste))
print(mean_squared_error(y_casas_teste, previsoes_teste))
print(np.sqrt(mean_squared_error(y_casas_teste, previsoes_teste)))
grafico1 = px.scatter(x=x_casas_teste.ravel(), y=y_casas_teste)
grafico2 = px.line(x=x_casas_teste.ravel(), y=previsoes_teste)
grafico2.data[0].line.color = "red"
grafico3 = go.Figure(data=grafico1.data + grafico2.data)
grafico3.show()
