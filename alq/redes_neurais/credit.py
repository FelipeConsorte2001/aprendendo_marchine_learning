import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )
rede_neural_credit = MLPClassifier(
    max_iter=1500,
    verbose=True,
    tol=0.0000100,
    solver="adam",
    activation="relu",
    hidden_layer_sizes=(2, 2),
)
rede_neural_credit.fit(x_credit_treinamento, y_credito_treinamento)
previsoes = rede_neural_credit.predict(x_credit_teste)
print(accuracy_score(y_credit_teste, previsoes))
cm = ConfusionMatrix(rede_neural_credit)
cm.fit(x_credit_treinamento, y_credito_treinamento)
cm.score(x_credit_teste, y_credit_teste)
cm.show()
print(classification_report(y_credit_teste, previsoes))
