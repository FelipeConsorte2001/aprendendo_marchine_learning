import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.under_sampling import TomekLinks
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

base_census = pd.read_csv("../../census.csv")
x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values
print(x_census, y_census)
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
tl = TomekLinks(sampling_strategy="majority")
x_under, y_under = tl.fit_resample(x_census, y_census)
print(x_under.shape, y_under.shape)
onehotencoder_census = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
    remainder="passthrough",
)
x_census = onehotencoder_census.fit_transform(x_census).toarray()
print(x_census[0])
(
    X_census_treinamento_under,
    X_census_teste_under,
    y_census_treinamento_under,
    y_census_teste_under,
) = train_test_split(x_under, y_under, test_size=0.15, random_state=0)
X_census_treinamento_under.shape, X_census_teste_under.shape
random_forest_census = RandomForestClassifier(
    criterion="entropy", min_samples_leaf=1, min_samples_split=5, n_estimators=100
)
random_forest_census.fit(X_census_treinamento_under, y_census_treinamento_under)
previsoes = random_forest_census.predict(X_census_teste_under)
print(accuracy_score(y_census_teste_under, previsoes))
