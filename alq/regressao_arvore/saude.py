import plotly.express as px
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np

base_plano_saude2 = pd.read_csv("./../../database/plano_saude.csv")
print(base_plano_saude2)
x_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values
regressor_arvire_saude = DecisionTreeRegressor()
regressor_arvire_saude.fit(x_plano_saude2, y_plano_saude2)
previsoes = regressor_arvire_saude.predict(x_plano_saude2)
print(previsoes)
print(regressor_arvire_saude.score(x_plano_saude2, y_plano_saude2))
grafico = px.scatter(x=x_plano_saude2.ravel(), y=y_plano_saude2)
grafico.add_scatter(x=x_plano_saude2.ravel(), y=previsoes, name="regressão")
grafico.show()

x_teste_arvore = np.arange(min(x_plano_saude2), max(x_plano_saude2), 0.1)
print(x_teste_arvore)
x_teste_arvore = x_teste_arvore.reshape(-1, 1)
grafico = px.scatter(x=x_plano_saude2.ravel(), y=y_plano_saude2)
grafico.add_scatter(
    x=x_teste_arvore.ravel(),
    y=regressor_arvire_saude.predict(x_teste_arvore),
    name="regressão",
)
grafico.show()
