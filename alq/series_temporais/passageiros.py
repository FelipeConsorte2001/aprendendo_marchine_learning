import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

database_path = "./../../database/AirPassengers.csv"
dataset = pd.read_csv(database_path)
dateparse = pd.to_datetime(dataset["Month"], format="%Y-%m")
dataset = pd.read_csv(
    database_path, parse_dates=["Month"], index_col="Month", date_format=dateparse
)
time_series = dataset["#Passengers"]
print(time_series)
print(time_series[1])
print(time_series["1949-02"])
print(time_series["1950-01-01":"1950-07-31"])
print(time_series[:"1950-07-31"])
print(time_series.index.max())
print(time_series.index.min())
plt.plot(time_series)
plt.show()
time_series_datas = time_series["1960-01-01":"1960-12-01"]
plt.plot(time_series_datas)
plt.show()
