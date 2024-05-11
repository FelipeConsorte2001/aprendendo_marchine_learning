import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )

logistic_credit = LogisticRegression(random_state=1)
logistic_credit.fit(x_credit_treinamento, y_credito_treinamento)

print(logistic_credit.intercept_)
print(logistic_credit.coef_)
previsoes = logistic_credit.predict(x_credit_teste)
print(previsoes)

print(accuracy_score(y_credit_teste, previsoes))
cm = ConfusionMatrix(logistic_credit)
cm.fit(x_credit_treinamento, y_credito_treinamento)
cm.score(x_credit_teste, y_credit_teste)
cm.show()
print(classification_report(y_credit_teste, previsoes))
