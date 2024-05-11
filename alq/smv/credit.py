import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )
svm_credit = SVC(kernel="rbf", random_state=1, C=2.0)
svm_credit.fit(x_credit_treinamento, y_credito_treinamento)
previsoes = svm_credit.predict(x_credit_teste)
print(previsoes)
print(accuracy_score(y_credit_teste, previsoes))
cm = ConfusionMatrix(svm_credit)
cm.fit(x_credit_treinamento, y_credito_treinamento)
cm.score(x_credit_teste, y_credit_teste)
cm.show()
