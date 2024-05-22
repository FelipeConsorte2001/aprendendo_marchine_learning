import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

with open("../../census.pkl", "rb") as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = (
        pickle.load(f)
    )
rede_neural_census = MLPClassifier(
    max_iter=1500,
    verbose=True,
    tol=0.000010,
    solver="adam",
    activation="relu",
    hidden_layer_sizes=(55, 55),
)
rede_neural_census.fit(x_census_treinamento, y_census_treinamento)
previsoes = rede_neural_census.predict(x_census_teste)
print(accuracy_score(y_census_teste, previsoes))
cm = ConfusionMatrix(rede_neural_census)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)
cm.show()
print(classification_report(y_census_teste, previsoes))
