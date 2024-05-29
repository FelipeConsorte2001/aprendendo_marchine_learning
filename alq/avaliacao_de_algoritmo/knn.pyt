from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
y_credit = np.concatenate((y_credito_treinamento, y_credit_teste), axis=0)


parametros = {"n_neighbors": [3, 5, 10, 20], "p": [1, 2]}
grid_serach = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros)
grid_serach.fit(x_credit, y_credit)
melhores_parametros = grid_serach.best_params_
melhores_resultado = grid_serach.best_score_
print(melhores_parametros, melhores_resultado)
