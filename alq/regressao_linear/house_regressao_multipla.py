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

x_casas = base_casas.iloc[:, 2:18].values
print(x_casas)


y_casas = base_casas.iloc[:, 1].values
print(y_casas)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = (
    train_test_split(x_casas, y_casas, test_size=0.3, random_state=0)
)
regresso_multiplo_casas = LinearRegression()
regresso_multiplo_casas.fit(x_casas_treinamento, y_casas_treinamento)
print(regresso_multiplo_casas.intercept_)
print(regresso_multiplo_casas.coef_)
print(regresso_multiplo_casas.score(x_casas_treinamento, y_casas_treinamento))
print(regresso_multiplo_casas.score(x_casas_teste, y_casas_teste))
previsoes = regresso_multiplo_casas.predict(x_casas_teste)
print(previsoes)
print(mean_absolute_error(y_casas_teste, previsoes))
print(mean_squared_error(y_casas_teste, previsoes))
