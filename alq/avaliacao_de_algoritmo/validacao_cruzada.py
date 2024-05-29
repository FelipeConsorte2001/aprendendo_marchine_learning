from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import pandas as pd

with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
y_credit = np.concatenate((y_credito_treinamento, y_credit_teste), axis=0)


resultados_arvore = []
resultados_random_forest = []
resultados_knn = []
resultados_logistica = []
resultados_svm = []
resultados_rede_neural = []


for i in range(30):
    kFold = KFold(n_splits=10, shuffle=True, random_state=i)
    arvore = DecisionTreeClassifier(
        criterion="entropy", min_samples_leaf=1, min_samples_split=5, splitter="best"
    )
    scores = cross_val_score(arvore, x_credit, y_credit, cv=kFold)
    resultados_arvore.append(scores.mean())

    random_forest = RandomForestClassifier(
        criterion="entropy", min_samples_leaf=1, min_samples_split=5, n_estimators=10
    )
    scores = cross_val_score(random_forest, x_credit, y_credit, cv=kFold)
    resultados_random_forest.append(scores.mean())

    knn = KNeighborsClassifier()
    scores = cross_val_score(knn, x_credit, y_credit, cv=kFold)
    resultados_knn.append(scores.mean())

    logistica = LogisticRegression(C=1.0, solver="lbfgs", tol=0.0001)
    scores = cross_val_score(logistica, x_credit, y_credit, cv=kFold)
    resultados_logistica.append(scores.mean())

    svm = SVC(kernel="rbf", C=2.0)
    scores = cross_val_score(svm, x_credit, y_credit, cv=kFold)
    resultados_svm.append(scores.mean())

    rede_neural = MLPClassifier(activation="relu", batch_size=56, solver="adam")
    scores = cross_val_score(rede_neural, x_credit, y_credit, cv=kFold)
    resultados_rede_neural.append(scores.mean())

print(resultados_arvore)
resultados = pd.DataFrame(
    {
        "Arvore": resultados_arvore,
        "Random forest": resultados_random_forest,
        "KNN": resultados_knn,
        "Logistica": resultados_logistica,
        "SVM": resultados_svm,
        "Rede neural": resultados_rede_neural,
    }
)
print(resultados)
print(resultados.describe)
print(resultados.var())
print((resultados.std() / resultados.mean()) / 100)
