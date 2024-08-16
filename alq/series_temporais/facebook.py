from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from prophet.plot import plot_components_plotly, plot_plotly

dataset = pd.read_csv("./../../database/page_wikipedia.csv")
dataset = dataset[["date", "views"]].rename(columns={"date": "ds", "views": "y"})
print(dataset)
dataset = dataset.sort_values(by="ds")
model = Prophet()
model.fit(dataset)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
print(forecast.tail(90))
model.plot(forecast, xlabel="Date", ylabel="Views")
plt.show()
model.plot_components(forecast)
plt.show()
grafico = plot_plotly(model, forecast)
grafico.show()
grafico2 = plot_components_plotly(model, forecast)
grafico2.show()
