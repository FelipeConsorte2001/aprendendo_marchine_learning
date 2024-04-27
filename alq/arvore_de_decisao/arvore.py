from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with open("./../../risco_credito.pkl", "rb") as f:
    x_risco_credito, y_risco_credito = pickle.load(f)

arvore_risco_credito = DecisionTreeClassifier(criterion="entropy")
arvore_risco_credito.fit(x_risco_credito, y_risco_credito)

print(arvore_risco_credito.feature_importances_)
previsores = ["história", "dívida", "garantias", "renda"]
figura, eixos = plt.subplots(nrows=1, ncols=1)
tree.plot_tree(
    arvore_risco_credito,
    feature_names=previsores,
    class_names=arvore_risco_credito.classes_,
    filled=True,
)
previsoes = arvore_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(previsoes)


with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )
print(x_credit_treinamento.shape, y_credito_treinamento.shape)
arvore_credit = DecisionTreeClassifier(criterion="entropy", random_state=0)
arvore_credit.fit(x_credit_treinamento, y_credito_treinamento)
previsores_credit = arvore_credit.predict(x_credit_teste)
print(previsores_credit)
print(accuracy_score(y_credit_teste, previsores_credit))
previsores = ["income", "age", "loan"]
fig, axes = plt.subplots(nrows=1, ncols=1)
tree.plot_tree(
    arvore_credit, feature_names=previsores, class_names=["0", "1"], filled=True
)
plt.show()

with open("../../census.pkl", "rb") as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = (
        pickle.load(f)
    )

arvore_census = DecisionTreeClassifier(criterion="entropy", random_state=0)
arvore_census.fit(x_census_treinamento, y_census_treinamento)
previsoes_census = arvore_census.predict(x_census_teste)
print(accuracy_score(y_census_teste, previsoes_census))
