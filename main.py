import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from math import floor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

base_credit = pd.read_csv("./credit_data.csv")
# print(np.unique(base_credit['default'], return_counts=True))

# sns.countplot(x=base_credit['default'])
# plt.hist(x=base_credit['age'])
# plt.hist(x=base_credit['income'])
# plt.hist(x=base_credit["loan"])

graphic = px.scatter_matrix(
    base_credit, dimensions=["age", "income", "loan"], color="default"
)
# graphic.show()

# apagar a coluna no banco de dados

base_credit2 = base_credit.drop("age", axis=1)
print(base_credit2)

# apagar somenete os registro com valores inconsitentes
base_credit3 = base_credit.drop(base_credit[base_credit["age"] < 0].index)
print(base_credit3)


# preencher os valores inconsistentes manualmente
media = base_credit["age"][base_credit["age"] > 0].mean()
base_credit["age"].loc[base_credit["age"] < 0] = 40.92

print(base_credit.loc[base_credit["age"] < 0])


# tratamento de valores faltantes
print(base_credit.isnull().sum())
print(base_credit.loc[pd.isnull(base_credit["age"])])
base_credit["age"] = base_credit["age"].fillna(media)

print(base_credit.loc[base_credit["clientid"].isin([29, 31, 32])])


x_credit = base_credit.iloc[:, 1:4].values
print(x_credit)
y_credit = base_credit.iloc[:, 4].values
print(y_credit)
print(x_credit[:, 0].min())
print(x_credit[:, 0].max())

scaler_credit = StandardScaler()
x_credit = scaler_credit.fit_transform(x_credit)
print(x_credit[:, 0].min())
print(x_credit[:, 0].max())
base_census = pd.read_csv("./census.csv")
print(base_census)
# grafic = sns.countplot(x=base_census["income"])
# plt.hist(x=base_census["age"])
# plt.hist(x=base_census["education-num"])
# plt.hist(x=base_census["hour-per-week"])
# plt.show()

# grafico = px.treemap(base_census, path=["workclass", "age"])
# grafico.show()
# grafico = px.treemap(base_census, path=["occupation", "relationship", "age"])

# grafico = px.parallel_categories(base_census, dimensions=["occupation", "relationship"])
# grafico = px.parallel_categories(
#     base_census, dimensions=["workclass", "occupation", "income"]
# )
# grafico.show()

x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values
print(x_census[:, 1])

label_encoder_teste = LabelEncoder()
teste = label_encoder_teste.fit_transform(x_census[:, 1])
print(teste)
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_rece = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_rece.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

onehotencoder_census = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
    remainder="passthrough",
)
x_census = onehotencoder_census.fit_transform(x_census).toarray()
print(x_census[0])

scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)
print(x_census.shape)

x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = (
    train_test_split(x_credit, y_credit, test_size=0.25, random_state=0)
)
print(x_credit_treinamento.shape)


x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = (
    train_test_split(x_census, y_census, test_size=0.15, random_state=0)
)
with open("credit.pkl", mode="wb") as f:
    pickle.dump(
        [x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f
    )

with open("census.pkl", mode="wb") as f:
    pickle.dump(
        [x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste], f
    )
