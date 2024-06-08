import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

base_casas = pd.read_csv("./../../database/house_prices.csv")
base_casas.drop("date", axis="columns", inplace=True)
x_casas = base_casas.iloc[:, 2:18].values
y_casas = base_casas.iloc[:, 1].values
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = (
    train_test_split(x_casas, y_casas, test_size=0.3, random_state=0)
)
print(x_casas_treinamento.shape)
regressor_arvore_casas = DecisionTreeRegressor()
regressor_arvore_casas.fit(x_casas_treinamento, y_casas_treinamento)
print(regressor_arvore_casas.score(x_casas_treinamento, y_casas_treinamento))
regressor_arvore_casas.fit(x_casas_treinamento, y_casas_treinamento)
print(regressor_arvore_casas.score(x_casas_teste, y_casas_teste))

previsoes = regressor_arvore_casas.predict(x_casas_teste)
print(mean_absolute_error(y_casas_teste, previsoes))
