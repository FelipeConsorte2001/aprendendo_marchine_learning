import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

base_casas = pd.read_csv("./../../database/house_prices.csv")
base_casas.drop("date", axis="columns", inplace=True)
x_casas = base_casas.iloc[:, 2:18].values
y_casas = base_casas.iloc[:, 1].values
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = (
    train_test_split(x_casas, y_casas, test_size=0.3, random_state=0)
)
scaler_x_casas = StandardScaler()
x_casas_treinamento_scaled = scaler_x_casas.fit_transform(x_casas_treinamento)
scaler_y_casas = StandardScaler()
y_casas_treinamento_scaled = scaler_y_casas.fit_transform(
    y_casas_treinamento.reshape(-1, 1)
)

x_casas_teste_scaled = scaler_x_casas.transform(x_casas_teste)
y_casas_teste_scaled = scaler_y_casas.transform(y_casas_teste.reshape(-1, 1))

regressor_svr_casas = SVR(kernel="rbf")
regressor_svr_casas.fit(x_casas_treinamento_scaled, y_casas_treinamento_scaled.ravel())
print(x_casas_teste_scaled.shape, y_casas_teste_scaled.shape)
print(regressor_svr_casas.score(x_casas_treinamento_scaled, y_casas_treinamento_scaled))
print(regressor_svr_casas.score(x_casas_teste_scaled, y_casas_teste_scaled))
previsoes = regressor_svr_casas.predict(x_casas_teste_scaled).reshape(-1, 1)
y_casa_teste_inverse = scaler_y_casas.inverse_transform(y_casas_teste_scaled)
previsoes_inverse = scaler_y_casas.inverse_transform(previsoes)
print(mean_absolute_error(y_casa_teste_inverse, previsoes_inverse))
