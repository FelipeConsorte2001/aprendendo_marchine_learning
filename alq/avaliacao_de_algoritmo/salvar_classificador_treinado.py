from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle
import numpy as np

with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
y_credit = np.concatenate((y_credito_treinamento, y_credit_teste), axis=0)

classificador_rede_neural = MLPClassifier(
    activation="relu", batch_size=56, solver="adam"
)
classificador_rede_neural.fit(x_credit, y_credit)
classificador_arvore = DecisionTreeClassifier(
    criterion="entropy", min_samples_leaf=1, min_samples_split=5, splitter="best"
)
classificador_arvore.fit(x_credit, y_credit)
classificador_svm = SVC(C=2.0, kernel="rbf", probability=True)
classificador_svm.fit(x_credit, y_credit)
pickle.dump(
    classificador_rede_neural,
    open("../../dump_finalizado/rede_neural_finalizado.sav", "wb"),
)
pickle.dump(
    classificador_arvore,
    open("../../dump_finalizado/arvore_finalizado.sav", "wb"),
)
pickle.dump(
    classificador_svm,
    open("../../dump_finalizado/svm_finalizado.sav", "wb"),
)
