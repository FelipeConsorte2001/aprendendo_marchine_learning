import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix

base_risco_credito = pd.read_csv("../../database/risco_credito.csv")

x_risco_credito = base_risco_credito.iloc[:, 0:4].values
y_risco_credito = base_risco_credito.iloc[:, 4].values

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()


x_risco_credito[:, 0] = label_encoder_historia.fit_transform(x_risco_credito[:, 0])
x_risco_credito[:, 1] = label_encoder_divida.fit_transform(x_risco_credito[:, 1])
x_risco_credito[:, 2] = label_encoder_garantia.fit_transform(x_risco_credito[:, 2])
x_risco_credito[:, 3] = label_encoder_renda.fit_transform(x_risco_credito[:, 3])

with open("./../../risco_credito.pkl", "wb") as f:
    pickle.dump([x_risco_credito, y_risco_credito], f)

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(x_risco_credito, y_risco_credito)
previsao = naive_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )


naive_credit_data = GaussianNB()
naive_credit_data.fit(x_credit_teste, y_credit_teste)
previsoes = naive_credit_data.predict(x_credit_teste)
acuracia = accuracy_score(y_credit_teste, previsoes)
print(acuracia)
confusao = confusion_matrix(y_credit_teste, previsoes)
print(confusao)

# cm = ConfusionMatrix(naive_credit_data)
# cm.fit(x_credit_treinamento, y_credito_treinamento)
# cm.score(x_credit_teste, y_credit_teste)
# cm.show()
# print(matrix)
print(classification_report(y_credit_teste, previsoes))


with open("../../census.pkl", "rb") as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = (
        pickle.load(f)
    )
print(x_census_treinamento.shape, y_census_treinamento.shape)
naive_census = GaussianNB()
naive_census.fit(x_census_treinamento, y_census_treinamento)
previsoes_census = naive_census.predict(x_census_teste)
# print(previsoes_census)
# print(y_census_teste)
# print(accuracy_score(y_census_teste, previsoes_census))
cm_census = ConfusionMatrix(naive_census)
cm_census.fit(x_census_treinamento, y_census_treinamento)
cm_census.score(x_census_teste, y_census_teste)
cm_census.show()
print(classification_report(y_census_teste, previsoes_census))
