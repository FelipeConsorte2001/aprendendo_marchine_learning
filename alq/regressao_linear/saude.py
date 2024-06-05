import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

base_plano = pd.read_csv("./../../database/plano_saude.csv")
print(base_plano)
x_plano_saude = base_plano.iloc[:, 0].values
print(x_plano_saude)
y_plano_saude = base_plano.iloc[:, 1].values
print(y_plano_saude)
print(np.corrcoef(x_plano_saude, y_plano_saude))
x_plano_saude = x_plano_saude.reshape(-1, 1)
regressor_plano_saude = LinearRegression()
regressor_plano_saude.fit(x_plano_saude, y_plano_saude)
print(regressor_plano_saude.intercept_)
print(regressor_plano_saude.coef_)
previsoes = regressor_plano_saude.predict(x_plano_saude)
print(previsoes)
x_plano_saude = x_plano_saude.ravel()
grafico = px.scatter(x=x_plano_saude, y=y_plano_saude)
grafico.add_scatter(x=x_plano_saude, y=previsoes, name="regressão")
grafico.show()
x_plano_saude = x_plano_saude.reshape(-1, 1)
print(regressor_plano_saude.score(x_plano_saude, y_plano_saude))
visualizador = ResidualsPlot(regressor_plano_saude)
visualizador.fit(x_plano_saude, y_plano_saude)
print(visualizador.poof())
