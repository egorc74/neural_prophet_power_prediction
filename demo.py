import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, set_log_level

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ipywidgets as widgets
from ipywidgets import interact_manual

# Set plotting backend to plotly for interactive plots and plotly-static for static plots
plotting_backend = "plotly-static"

df=pd.read_csv("data_15min_measurements.csv")

df["ID"] = df["ID"].astype(str)
IDs = df["ID"].unique()
df["ds"] = pd.to_datetime(df["ds"])

# use one year for faster training
df = df[df["ds"] > "2014-01-01"]

df.head()


fig = px.line(df[df["ds"] < "2014-02-01"], x="ds", y="y", color="ID")

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Energy consumption",
    legend_title="ID",
    height=600,
    width=1000,
)

if plotting_backend == "plotly-static":
    fig.show("svg")



df["summer"] = 0
df.loc[df["ds"].dt.month.isin([6, 7, 8]), "summer"] = 1
df["winter"] = 0
df.loc[df["ds"].dt.month.isin([12, 1, 2]), "winter"] = 1
df["fall"] = 0
df.loc[df["ds"].dt.month.isin([9, 10, 11]), "fall"] = 1
df["spring"] = 0
df.loc[df["ds"].dt.month.isin([3, 4, 5]), "spring"] = 1

# Conditional Seasonality: 4 Seasons, Weekday/Weekend distinction for each season, for daily seasonality
df["summer_weekday"] = 0
df.loc[(df["ds"].dt.month.isin([6, 7, 8])) & (df["ds"].dt.dayofweek.isin([0, 1, 2, 3, 4])), "summer_weekday"] = 1
df["summer_weekend"] = 0
df.loc[(df["ds"].dt.month.isin([6, 7, 8])) & (df["ds"].dt.dayofweek.isin([5, 6])), "summer_weekend"] = 1

df["winter_weekday"] = 0
df.loc[(df["ds"].dt.month.isin([12, 1, 2])) & (df["ds"].dt.dayofweek.isin([0, 1, 2, 3, 4])), "winter_weekday"] = 1
df["winter_weekend"] = 0
df.loc[(df["ds"].dt.month.isin([12, 1, 2])) & (df["ds"].dt.dayofweek.isin([5, 6])), "winter_weekend"] = 1

df["spring_weekday"] = 0
df.loc[(df["ds"].dt.month.isin([3, 4, 5])) & (df["ds"].dt.dayofweek.isin([0, 1, 2, 3, 4])), "spring_weekday"] = 1
df["spring_weekend"] = 0
df.loc[(df["ds"].dt.month.isin([3, 4, 5])) & (df["ds"].dt.dayofweek.isin([5, 6])), "spring_weekend"] = 1

df["fall_weekday"] = 0
df.loc[(df["ds"].dt.month.isin([9, 10, 11])) & (df["ds"].dt.dayofweek.isin([0, 1, 2, 3, 4])), "fall_weekday"] = 1
df["fall_weekend"] = 0
df.loc[(df["ds"].dt.month.isin([9, 10, 11])) & (df["ds"].dt.dayofweek.isin([5, 6])), "fall_weekend"] = 1

df.head()



quantiles = [0.015, 0.985]

params = {
    "n_lags": 7 * 24,
    "n_forecasts": 24,
    "n_changepoints": 20,
    "learning_rate": 0.01,
    "ar_layers": [32, 16, 16, 32],
    "yearly_seasonality": 10,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "epochs": 40,
    "batch_size": 1024,
    "quantiles": quantiles,
}


m = NeuralProphet(**params)
m.set_plotting_backend(plotting_backend)
set_log_level("ERROR")



m.add_seasonality(name="summer_weekly", period=7, fourier_order=14, condition_name="summer")
m.add_seasonality(name="winter_weekly", period=7, fourier_order=14, condition_name="winter")
m.add_seasonality(name="spring_weekly", period=7, fourier_order=14, condition_name="spring")
m.add_seasonality(name="fall_weekly", period=7, fourier_order=14, condition_name="fall")

m.add_seasonality(name="summer_weekday", period=1, fourier_order=6, condition_name="summer_weekday")
m.add_seasonality(name="winter_weekday", period=1, fourier_order=6, condition_name="winter_weekday")
m.add_seasonality(name="spring_weekday", period=1, fourier_order=6, condition_name="spring_weekday")
m.add_seasonality(name="fall_weekday", period=1, fourier_order=6, condition_name="fall_weekday")

m.add_seasonality(name="summer_weekend", period=1, fourier_order=6, condition_name="summer_weekend")
m.add_seasonality(name="winter_weekend", period=1, fourier_order=6, condition_name="winter_weekend")
m.add_seasonality(name="spring_weekend", period=1, fourier_order=6, condition_name="spring_weekend")
m.add_seasonality(name="fall_weekend", period=1, fourier_order=6, condition_name="fall_weekend")