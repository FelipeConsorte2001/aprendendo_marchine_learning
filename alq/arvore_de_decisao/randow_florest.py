from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix

with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )

random_forest_credit = RandomForestClassifier(
    n_estimators=40, criterion="entropy", random_state=0
)

random_forest_credit.fit(x_credit_treinamento, y_credito_treinamento)

previsoes = random_forest_credit.predict(x_credit_teste)
print(previsoes)
print(accuracy_score(y_credit_teste, previsoes))
cm_census = ConfusionMatrix(random_forest_credit)
cm_census.fit(x_credit_treinamento, y_credito_treinamento)
cm_census.score(x_credit_teste, y_credit_teste)
cm_census.show()

with open("../../census.pkl", "rb") as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = (
        pickle.load(f)
    )

random_forest_census = RandomForestClassifier(
    n_estimators=100, criterion="entropy", random_state=0
)

random_forest_census.fit(x_census_treinamento, y_census_treinamento)
previsoes_census = random_forest_census.predict(x_census_teste)
print("-----------", previsoes_census)
print(accuracy_score(y_census_teste, previsoes_census))
cm_census = ConfusionMatrix(random_forest_census)
cm_census.fit(x_census_treinamento, y_census_treinamento)
cm_census.score(x_census_teste, y_census_teste)
cm_census.show()
