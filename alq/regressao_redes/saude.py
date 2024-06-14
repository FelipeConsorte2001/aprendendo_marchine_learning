import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

base_plano_saude2 = pd.read_csv("./../../database/plano_saude2.csv")
print(base_plano_saude2)
x_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values
scaler_x = StandardScaler()
x_plano_saude2_scaled = scaler_x.fit_transform(x_plano_saude2)
scaler_y = StandardScaler()
y_plano_saude2_scaled = scaler_y.fit_transform(y_plano_saude2.reshape(-1, 1))
regressor_rna_saude = MLPRegressor(max_iter=1000)
regressor_rna_saude.fit(x_plano_saude2_scaled, y_plano_saude2_scaled.ravel())
print(regressor_rna_saude.score(x_plano_saude2_scaled, y_plano_saude2_scaled))
grafico = px.scatter(x=x_plano_saude2_scaled.ravel(), y=y_plano_saude2_scaled.ravel())
grafico.add_scatter(
    x=x_plano_saude2_scaled.ravel(),
    y=regressor_rna_saude.predict(x_plano_saude2_scaled),
    name="regressão",
)
grafico.show()
novo = [[40]]
novo = scaler_x.transform(novo)
print(scaler_y.inverse_transform([regressor_rna_saude.predict(novo)]))
