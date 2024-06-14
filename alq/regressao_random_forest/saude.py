import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

base_plano = pd.read_csv("./../../database/plano_saude.csv")
print(base_plano)
x_plano_saude = base_plano.iloc[:, 0].values
print(x_plano_saude)
y_plano_saude = base_plano.iloc[:, 1].values
x_plano_saude = x_plano_saude.reshape(-1, 1)
regressor_random_forest_saude = RandomForestRegressor(n_estimators=10)
regressor_random_forest_saude.fit(x_plano_saude, y_plano_saude)
print(regressor_random_forest_saude.score(x_plano_saude, y_plano_saude))
x_teste_arvore = np.arange(min(x_plano_saude), max(x_plano_saude), 0.1)
x_teste_arvore = x_teste_arvore.reshape(-1, 1)
grafico = px.scatter(x=x_plano_saude.ravel(), y=y_plano_saude)
grafico.add_scatter(
    x=x_teste_arvore.ravel(),
    y=regressor_random_forest_saude.predict(x_teste_arvore),
    name="regressão",
)
grafico.show()
print(regressor_random_forest_saude.predict([[40]]))
