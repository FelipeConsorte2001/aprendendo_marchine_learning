import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

base_census = pd.read_csv("../../census.csv")
x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values
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
scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)
x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = (
    train_test_split(x_census, y_census, test_size=0.15, random_state=0)
)
pca = PCA(n_components=8)
x_census_treinamento_pca = pca.fit_transform(x_census_treinamento)
x_census_teste_pca = pca.transform(x_census_teste)
print(x_census_treinamento_pca.shape, x_census_teste_pca.shape)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
random_forest_census_pca = RandomForestClassifier(n_estimators=40, random_state=0)
random_forest_census_pca.fit(x_census_treinamento_pca, y_census_treinamento)
previsoes = random_forest_census_pca.predict(x_census_teste_pca)
print(previsoes)
print(accuracy_score(y_census_teste, previsoes))
