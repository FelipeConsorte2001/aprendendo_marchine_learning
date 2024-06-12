import plotly.express as px
import pandas as pd
from sklearn.svm import SVR
import numpy as np
from sklearn.preprocessing import StandardScaler

base_plano_saude2 = pd.read_csv("./../../database/plano_saude2.csv")
print(base_plano_saude2)
x_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values

print(x_plano_saude2, y_plano_saude2)
regressor_svr_saude = SVR(kernel="linear")
regressor_svr_saude.fit(x_plano_saude2, y_plano_saude2)
grafico = px.scatter(x=x_plano_saude2.ravel(), y=y_plano_saude2)
grafico.add_scatter(
    x=x_plano_saude2.ravel(),
    y=regressor_svr_saude.predict(x_plano_saude2),
    name="regressão",
)
grafico.show()
regressor_svr_saude_poly = SVR(kernel="poly", degree=3)
regressor_svr_saude_poly.fit(x_plano_saude2, y_plano_saude2)
grafico2 = px.scatter(x=x_plano_saude2.ravel(), y=y_plano_saude2)
grafico2.add_scatter(
    x=x_plano_saude2.ravel(),
    y=regressor_svr_saude_poly.predict(x_plano_saude2),
    name="regressão",
)
grafico2.show()
# regressor_svr_saude_rbf = SVR(kernel="rbf")
# regressor_svr_saude_rbf.fit(x_plano_saude2, y_plano_saude2)

# grafico3 = px.scatter(x=x_plano_saude2.ravel(), y=y_plano_saude2)
# grafico3.add_scatter(
#     x=x_plano_saude2.ravel(),
#     y=regressor_svr_saude_rbf.predict(x_plano_saude2),
#     name="regressão",
# )
# grafico3.show()
scaler_x = StandardScaler()
x_plano_saude2_scaled = scaler_x.fit_transform(x_plano_saude2)
scaler_y = StandardScaler()
y_plano_saude2_scaled = scaler_y.fit_transform(y_plano_saude2.reshape(-1, 1))
regressor_svr_saude_rbf = SVR(kernel="rbf")
regressor_svr_saude_rbf.fit(x_plano_saude2_scaled, y_plano_saude2_scaled.ravel())
grafico3 = px.scatter(x=x_plano_saude2_scaled.ravel(), y=y_plano_saude2_scaled.ravel())
grafico3.add_scatter(
    x=x_plano_saude2_scaled.ravel(),
    y=regressor_svr_saude_rbf.predict(x_plano_saude2_scaled),
    name="regressão",
)
grafico3.show()
novo = [[40]]
novo = scaler_x.transform(novo)

print(regressor_svr_saude_rbf.predict(novo))
print(scaler_y.inverse_transform([regressor_svr_saude_rbf.predict(novo)]))
