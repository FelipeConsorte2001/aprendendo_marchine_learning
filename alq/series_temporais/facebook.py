from prophet import Prophet
import pandas as pd

dataset = pd.read_csv("./../../database/page_wikipedia.csv")
dataset = dataset[["date", "views"]].rename(columns={"date": "ds", "views": "y"})
print(dataset)
dataset = dataset.sort_values(by="ds")
