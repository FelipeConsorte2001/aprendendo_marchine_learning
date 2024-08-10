import pandas as pd
import plotly.express as px
from pyod.models.knn import KNN
import numpy as np

base_credit = pd.read_csv("./../../credit_data.csv")
base_credit.dropna(inplace=True)
detector = KNN()
detector.fit(base_credit.iloc[:, 1:4])
previsoes = detector.labels_
print(previsoes)
print(np.unique(previsoes, return_counts=True))
confianca_previsoes = detector.decision_scores_
print(confianca_previsoes)
outliers = []
for i in range(len(previsoes)):
    if previsoes[i] == 1:
        outliers.append(i)
print(outliers)
lista_outliers = base_credit.iloc[outliers, :]
print(lista_outliers)
