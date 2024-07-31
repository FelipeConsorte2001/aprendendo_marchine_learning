from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
smote = SMOTE(sampling_strategy="minority")
x_over, y_over = smote.fit_resample(x_census, y_census)
print(x_over.shape, y_over.shape)

onehotencorder = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
    remainder="passthrough",
)
X_census = onehotencorder.fit_transform(x_over).toarray()
print(X_census)

(
    X_census_treinamento_over,
    X_census_teste_over,
    y_census_treinamento_over,
    y_census_teste_over,
) = train_test_split(x_over, y_over, test_size=0.15, random_state=0)
random_forest_census = RandomForestClassifier(
    criterion="entropy", min_samples_leaf=1, min_samples_split=5, n_estimators=100
)
random_forest_census.fit(X_census_treinamento_over, y_census_treinamento_over)
previsoes = random_forest_census.predict(X_census_teste_over)
accuracy_score(y_census_teste_over, previsoes)
print(accuracy_score(y_census_teste_over, previsoes))
