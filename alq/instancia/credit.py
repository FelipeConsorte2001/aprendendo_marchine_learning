import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

with open("../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )
print(x_credit_treinamento.shape, y_credit_treinamento.shape)
knn_credit = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn_credit.fit(x_credit_treinamento, y_credit_treinamento)
previsoes = knn_credit.predict(x_credit_teste)
print(previsoes)
acuracia = accuracy_score(y_credit_teste, previsoes)
print(acuracia)
cm = ConfusionMatrix(knn_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
cm.show()
