from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

base_casas = pd.read_csv("./../../database/house_prices.csv")
base_casas.drop("date", axis="columns", inplace=True)
print(base_casas)
print(base_casas.describe)

x_casas = base_casas.iloc[:, 2:18].values
print(x_casas)


y_casas = base_casas.iloc[:, 1].values
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = (
    train_test_split(x_casas, y_casas, test_size=0.3, random_state=0)
)
poly = PolynomialFeatures(degree=2)
X_casas_treinamento_poly = poly.fit_transform(x_casas_treinamento)
X_casas_teste_poly = poly.transform(x_casas_teste)
regressor_casas_poly = LinearRegression()
regressor_casas_poly.fit(X_casas_treinamento_poly, y_casas_treinamento)
print(regressor_casas_poly.score(X_casas_treinamento_poly, y_casas_treinamento))
print(regressor_casas_poly.score(X_casas_teste_poly, y_casas_teste))
previsoes = regressor_casas_poly.predict(X_casas_teste_poly)
print(previsoes)
print(y_casas_teste)
print(mean_absolute_error(y_casas_teste, previsoes))
