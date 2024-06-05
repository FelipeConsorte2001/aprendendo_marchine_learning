import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

base_casas = pd.read_csv("./../../database/house_prices.csv")
base_casas.drop("date", axis="columns", inplace=True)
print(base_casas)
print(base_casas.describe)
print(base_casas.isnull().sum())
print(base_casas.corr())
figura = plt.figure(figsize=(20, 20))
sns.heatmap(base_casas.corr(), annot=True)
plt.show()
x_casas = base_casas.iloc[:, 4:5].values
print(x_casas)
y_casas = base_casas.iloc[:, 1].values
print(y_casas)
