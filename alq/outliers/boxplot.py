import pandas as pd
import plotly.express as px

base_credit = pd.read_csv("./../../credit_data.csv")
print(base_credit)
print(base_credit.isnull().sum())
base_credit.dropna(inplace=True)
print(base_credit.isnull().sum())
# outliers idade

grafico = px.box(base_credit, y="age")
grafico.show()
outliers_age = base_credit[base_credit["age"] < 0]
print(outliers_age)

# outlies loan
grafico = px.box(base_credit, y="loan")
grafico.show()

outliers_loan = base_credit[base_credit["loan"] >= 13300]
print(outliers_loan)
