import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier

base_census = pd.read_csv("../../census.csv")
colunas = base_census.columns[:-1]
x_census = base_census.iloc[:, 0:14].values
y_cesus = base_census.iloc[:, 14].values
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()
x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])
scaler = MinMaxScaler()
x_census_scaler = scaler.fit_transform(x_census)

for i in range(x_census.shape[1]):
    print(x_census_scaler[:, i].var())
selecao = VarianceThreshold(threshold=0.05)
x_census_variancia = selecao.fit_transform(x_census_scaler)
print(x_census_variancia.shape)
indices = np.where(selecao.variances_ > 0.5)
print(colunas[indices])
base_census_variancia = base_census.drop(
    columns=[
        "age",
        "workclass",
        "final-weight",
        "education-num",
        "race",
        "capital-gain",
        "capital-loos",
        "hour-per-week",
        "native-country",
    ],
    axis=1,
)
print(base_census_variancia)
x_census_variancia = base_census_variancia.iloc[:, 0:5].values
y_census_variancia = base_census_variancia.iloc[:, 5].values
print(y_census_variancia, x_census_variancia)
x_census_variancia[:, 0] = label_encoder_education.fit_transform(
    x_census_variancia[:, 0]
)
x_census_variancia[:, 1] = label_encoder_marital.fit_transform(x_census_variancia[:, 1])
x_census_variancia[:, 2] = label_encoder_occupation.fit_transform(
    x_census_variancia[:, 2]
)
x_census_variancia[:, 3] = label_encoder_relationship.fit_transform(
    x_census_variancia[:, 3]
)
x_census_variancia[:, 4] = label_encoder_sex.fit_transform(x_census_variancia[:, 4])
onehotencorder = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [0, 1, 2, 3, 4])], remainder="passthrough"
)
x_census_variancia = onehotencorder.fit_transform(x_census_variancia).toarray()
print(x_census_variancia)
(
    X_census_treinamento_var,
    X_census_teste_var,
    y_census_treinamento_var,
    y_census_teste_var,
) = train_test_split(
    x_census_variancia, y_census_variancia, test_size=0.15, random_state=0
)
print(X_census_treinamento_var.shape, X_census_teste_var.shape)
random_forest_var = RandomForestClassifier(
    criterion="entropy", min_samples_leaf=1, min_samples_split=5, n_estimators=100
)
random_forest_var.fit(X_census_treinamento_var, y_census_treinamento_var)

previsoes = random_forest_var.predict(X_census_teste_var)
print(accuracy_score(y_census_teste_var, previsoes))
selecao = ExtraTreesClassifier()
selecao.fit(x_census_scaler, y_cesus)
importancias = selecao.feature_importances_
print(importancias.sum())
indices = []
for i in range(len(importancias)):
    if importancias[i] >= 0.029:
        indices.append(i)

print(indices)
x_census_extra = x_census[:, indices]

onehotencorder = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7])], remainder="passthrough"
)
X_census_extra = onehotencorder.fit_transform(x_census_extra).toarray()
print(X_census_extra)
(
    X_census_treinamento_extra,
    X_census_teste_extra,
    y_census_treinamento_extra,
    y_census_teste_extra,
) = train_test_split(X_census_extra, y_cesus, test_size=0.15, random_state=0)
print(X_census_treinamento_extra.shape, X_census_teste_extra.shape)
random_forest_extra = RandomForestClassifier(
    criterion="entropy", min_samples_leaf=1, min_samples_split=5, n_estimators=100
)
random_forest_extra.fit(X_census_treinamento_extra, y_census_treinamento_extra)
previsoes = random_forest_extra.predict(X_census_teste_extra)
print(accuracy_score(y_census_teste_extra, previsoes))
