import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

with open("../../census.pkl", "rb") as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = (
        pickle.load(f)
    )


logistic_census = LogisticRegression(random_state=1)
logistic_census.fit(x_census_treinamento, y_census_treinamento)

previsoes = logistic_census.predict(x_census_teste)
print(previsoes)
print(accuracy_score(y_census_teste, previsoes))
cm_census = ConfusionMatrix(logistic_census)
cm_census.fit(x_census_treinamento, y_census_treinamento)
cm_census.score(x_census_teste, y_census_teste)
cm_census.show()
print(classification_report(y_census_teste, previsoes))
