import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

base_plano_saude2 = pd.read_csv("./../../database/plano_saude.csv")
print(base_plano_saude2)
x_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values
poly = PolynomialFeatures(degree=2)
x_plano_saude2_poly = poly.fit_transform(x_plano_saude2)
regressor_saude_polinomial = LinearRegression()
regressor_saude_polinomial.fit(x_plano_saude2_poly, y_plano_saude2)

# b0
print(regressor_saude_polinomial.intercept_)
# b1 (n)
print(regressor_saude_polinomial.coef_)
novo = poly.transform([[40]])
previsao_novo = regressor_saude_polinomial.predict(novo)
print(previsao_novo)
previsao = regressor_saude_polinomial.predict(x_plano_saude2_poly)
print(previsao)
grafico = px.scatter(x=x_plano_saude2[:, 0], y=y_plano_saude2)
grafico.add_scatter(x=x_plano_saude2[:, 0], y=previsao, name="regressão")
grafico.show()
