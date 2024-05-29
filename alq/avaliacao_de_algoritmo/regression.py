from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
y_credit = np.concatenate((y_credito_treinamento, y_credit_teste), axis=0)


parametros = {
    "tol": [0.0001, 0.00001, 0.000001],
    "C": [1.0, 1.5, 2.0],
    "solver": ["lbfgs", "sag", "saga"],
}
grid_serach = GridSearchCV(estimator=LogisticRegression(), param_grid=parametros)
grid_serach.fit(x_credit, y_credit)
melhores_parametros = grid_serach.best_params_
melhores_resultado = grid_serach.best_score_
print(melhores_parametros, melhores_resultado)
