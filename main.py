from neuralprophet import NeuralProphet
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from tkinter import scrolledtext
from tqdm import tqdm


           # adjust
DT_FORMAT     = None                    # e.g. "%d.%m.%Y %H:%M:%S" or just None
FREQ          = "15min"                 # <- NeuralProphet frequency string
FORECAST_HORIZON = 96 




class Prediction:
    def __init__(self,data_name,prediction_collumn_name,timestamp_collumn_name):
        df=pd.read_csv(f"{data_name}")
        self.prediction_collumn_name=prediction_collumn_name
        self.timestamp_collumn_name=timestamp_collumn_name
        self.graph=False
        self.rmse_mean=0
        
        df.rename(columns={f"{self.timestamp_collumn_name}":"ds",f"{self.prediction_collumn_name}":"y"},inplace=True)
        df['ds'] = pd.to_datetime(df['ds'], format="%d. %m. %Y %H:%M:%S")
        
        # Then fill missing values explicitly
        df['y'] = df['y'].interpolate()  # or .fillna(method='ffill')
        df.sort_values("ds", inplace=True)
        df = df[['ds', 'y']]
        df=self.add_flags(df)  #add weekend days weekdays and seasons
        self.df=df



    def add_flags(self, _df: pd.DataFrame) -> pd.DataFrame:
        d = _df.copy()
        # season flags
        d["spring"] = d["ds"].dt.month.isin([3, 4, 5]).astype(int)
        d["summer"] = d["ds"].dt.month.isin([6, 7, 8]).astype(int)
        d["fall"]   = d["ds"].dt.month.isin([9,10,11]).astype(int)  # autumn
        d["winter"] = d["ds"].dt.month.isin([12,1,2]).astype(int)
        # weekday / weekend flags
        d["weekday"] = (d["ds"].dt.dayofweek < 5).astype(int)
        d["weekend"] = (d["ds"].dt.dayofweek >= 5).astype(int)
        return d
  
    def make_model(self):   #creating neural prophet model
        self.m = NeuralProphet(
            n_lags       = 96*2,         #future prediction in min
            n_forecasts  = 96,
            yearly_seasonality = False, 
            weekly_seasonality = False,
            daily_seasonality  = False,
            epochs=40
        )
        # seasonality per season (weekly because period=7 days)
        for s in ["spring", "summer", "fall", "winter"]:
            self.m.add_seasonality(name=f"{s}_weekly", period=7, fourier_order=10,
                            condition_name=s)
        # weekday vs weekend modifiers (also weekly period)
        self.m.add_seasonality(name="weekday_effect", period=7, fourier_order=4,
                        condition_name="weekday")
        self.m.add_seasonality(name="weekend_effect", period=7, fourier_order=4,
                        condition_name="weekend")
        
    def day_rolling_prediction(self,end_day):
        start_day=self.df["ds"].dt.floor("D").min() + pd.Timedelta(days=7)    

        self.daily_rmse    = []

        train_df = self.df[(self.df["ds"] < end_day)] #data before the day
        train_df.to_csv("train.csv", index=False)

        test_df  = self.df[(self.df["ds"] >= end_day) & (self.df["ds"] < end_day + pd.Timedelta(days=1))]
        self.make_model()
        self.m.fit(train_df, freq=FREQ, progress="off", minimal=True)
        
        future = self.m.make_future_dataframe(train_df, periods=FORECAST_HORIZON, n_historic_predictions=False)
        # make_future_dataframe does *not* know our flags → add them now
        future = self.add_flags(future)
        print(f"future tail{future.tail()}")
        
        
        fcst = self.m.predict(future)
        fcst.to_csv("fcst.csv", index=False)

        # fcst_day = fcst[["ds", "yhat1"]].copy()            # yhat1 only (point forecast)
        fcst_day=self.prepare_forecast_data(fcst,day=end_day)
        fcst_day.to_csv("forecast_day.csv", index=False)

        
        # Error (RMSE) ------------------------------------------------------
        merged = fcst_day.merge(test_df[["ds", "y"]], on="ds")
        rmse   = np.sqrt(np.mean((merged["y"] - merged["prediction"]) ** 2))
        # self.daily_rmse.append((day, rmse))           
        # rmse_df = pd.DataFrame(self.daily_rmse, columns=["day", "RMSE"])
        mean_actual = np.mean(merged['y'])
        rmse_percent = (rmse / mean_actual) * 100
        print(f"RMSE (% of mean): {rmse_percent:.2f}%")

        # print("Mean RMSE over all rolling steps:", rmse_df["RMSE"].mean())
        print(fcst_day.tail())
        self.plot_graph(actual_day=test_df,forecast_day=fcst_day)
        plt.show()


    def plot_graph(self,actual_day,forecast_day):
        if not self.graph:
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title(f"Forecast vs Actual on ")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.xticks(rotation=45)
            self.graph=True
        plt.plot(actual_day['ds'], actual_day['y'], label='Actual', linewidth=2)
        plt.plot(forecast_day['ds'], forecast_day['prediction'], label='Forecast', linestyle='--', linewidth=2)

    def prepare_forecast_data(self,forecast,day):
        yhat_cols = [f"yhat{i}" for i in range(1, 96 + 1)]
        df = forecast[['ds'] + yhat_cols].copy()
        df['ds'] = pd.to_datetime(df['ds'])

        df=df[df['ds']>=day]                       #comment out after testing
        df["prediction"]=0
        i=1
        for idx,row in df.iterrows():
            df.at[idx, 'prediction'] = row[f"yhat{i}"]
            i=i+1

        df_final = df[["ds", "prediction"]]
        return df_final

    def month_prediction(self,month,number_of_days):
        day=month
        while day!=month+pd.Timedelta(days=number_of_days):
            self.day_prediction(day=day)
            day=day+pd.Timedelta(days=1)
        print(self.rmse_mean/number_of_days)
        plt.show()





p=Prediction(data_name="data_15min_measurements.csv",prediction_collumn_name="P+ Prejeta delovna moč",timestamp_collumn_name="Časovna značka")
day = pd.Timestamp("2023-04-20 00:00:00")
p.day_rolling_prediction(day)