# prediction.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATE_FMT = "%d. %m. %Y %H:%M:%S"      # matches “01. 05. 2023 00:00:00”
FREQ     = "15min"                    # your data resolution
N_LAGS   = 7 * 24 * 3                 # 1 week of 15-min lags = 672
N_FCST   = 96                         # 24 h * 4 points / h
SEAS_FOURIER = 14                     # keeps seasonality smooth


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Predictor:
    csv_path: str
    target_col: str
    ts_col: str
    df: pd.DataFrame = field(init=False)
    rmse_scores: List[float] = field(default_factory=list, init=False)

    # model hyper-parameters as kwargs (easier to tune later)
    model_params: dict = field(default_factory=lambda: {
        "n_lags": N_LAGS,
        "n_forecasts": N_FCST,
        "n_changepoints": 20,
        "learning_rate": 0.01,
        "ar_layers": [32, 16, 16, 32],
        "yearly_seasonality": 10,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "epochs": 40,
        "batch_size": 1024,
        "quantiles": [0.015, 0.985],
    })

    # ── INITIALISATION ────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Load CSV, rename columns and add conditional seasonality flags."""
        self.df = (
            pd.read_csv(self.csv_path)
              .rename(columns={self.ts_col: "ds", self.target_col: "y"})
              .assign(ds=lambda d: pd.to_datetime(d["ds"], format=DATE_FMT))
              .loc[:, ["ds", "y"]]                          # keep only two cols
              .sort_values("ds")
              .reset_index(drop=True)
        )
        self._add_season_flags()                            # adds 8 binary cols

    # ── PUBLIC API ────────────────────────────────────────────────────────
    def predict_month(self, start_day: str, days: int = 10) -> None:
        """Rolling-origin evaluation day-by-day."""
        start = pd.Timestamp(start_day)
        for offset in range(days):
            day = start + pd.Timedelta(days=offset)
            self._predict_single_day(day)

        # summary
        if self.rmse_scores:
            print(f"\nAvg RMSE over {days} days: {np.mean(self.rmse_scores):.4f}")
        plt.tight_layout(); plt.show()

    # ── INTERNALS ────────────────────────────────────────────────────────
    def _add_season_flags(self) -> None:
        """Vectorised creation of seasonal and weekday/weekend flags."""
        month  = self.df["ds"].dt.month
        wday   = self.df["ds"].dt.dayofweek   # Mon=0 … Sun=6

        def flag(mask: pd.Series, name: str) -> pd.Series:
            return mask.astype("int8").rename(name)

        self.df = self.df.join(pd.concat([
            flag(month.isin([ 6, 7, 8]),              "summer"),
            flag(month.isin([12, 1, 2]),              "winter"),
            flag(month.isin([ 9,10,11]),              "fall"),
            flag(month.isin([ 3, 4, 5]),              "spring"),

            flag((month.isin([ 6,7,8]) ) & (wday<5),  "summer_weekday"),
            flag((month.isin([ 6,7,8]) ) & (wday>4),  "summer_weekend"),
            flag((month.isin([12,1,2]) ) & (wday<5),  "winter_weekday"),
            flag((month.isin([12,1,2]) ) & (wday>4),  "winter_weekend"),
            flag((month.isin([ 3,4,5]) ) & (wday<5),  "spring_weekday"),
            flag((month.isin([ 3,4,5]) ) & (wday>4),  "spring_weekend"),
            flag((month.isin([ 9,10,11])) & (wday<5), "fall_weekday"),
            flag((month.isin([ 9,10,11])) & (wday>4), "fall_weekend"),
        ], axis=1))

    def _new_model(self) -> NeuralProphet:
        """Fresh NeuralProphet instance with all conditional seasonalities."""
        m = NeuralProphet(**self.model_params)

        # weekly seasonality per season
        for season in ["summer", "winter", "spring", "fall"]:
            m.add_seasonality(
                name=f"{season}_weekly", period=7, fourier_order=SEAS_FOURIER,
                condition_name=season)

        # weekday/weekend daily pattern
        for tag in ["summer", "winter", "spring", "fall"]:
            m.add_seasonality(
                name=f"{tag}_weekday", period=1, fourier_order=6,
                condition_name=f"{tag}_weekday")
            m.add_seasonality(
                name=f"{tag}_weekend", period=1, fourier_order=6,
                condition_name=f"{tag}_weekend")
        return m

    def _predict_single_day(self, day: pd.Timestamp) -> None:
        """Train up to *day*-1 and forecast the next 24 h."""
        train_df = self.df[self.df["ds"] < day]
        model    = self._new_model()
        model.fit(train_df, freq=FREQ, progress="none")

        # Forecast 24 h (96×15-min) and slice exactly that day
        future      = model.make_future_dataframe(train_df, periods=N_FCST)
        forecast    = model.predict(future)
        fcst_slice  = forecast.query("@day <= ds < @day + @pd.Timedelta('1D')")
        actual      = self.df.query("@day <= ds < @day + @pd.Timedelta('1D')")

        self._plot_day(actual, fcst_slice, day)
        self._update_rmse(actual, fcst_slice, day)

    def rolling_daily_forecast(
            self,
            start_day: str,
            horizon_days: int = 10,
    ) -> None:
        """
        Train on *all* data strictly before each day, forecast the next 24 h,
        do it `horizon_days` times in a row, and collect RMSEs/plots.
        """
        start = pd.Timestamp(start_day)

        for step in range(horizon_days):
            day = start + pd.Timedelta(days=step)

            # ── 1.  train up-to-yesterday ────────────────────────────
            train_df = self.df[self.df["ds"] < day]

            model = self._new_model()
            model.fit(train_df, freq=FREQ, progress="none")      # silent training

            # ── 2.  forecast the next 24 h (96×15-min) ───────────────
            future   = model.make_future_dataframe(train_df, periods=N_FCST)
            forecast = model.predict(future)

            fcst_day = forecast.query("@day <= ds < @day + '1D'")
            act_day  = self.df.query("@day <= ds < @day + '1D'")

            self._plot_day(act_day, fcst_day, day)
            self._update_rmse(act_day, fcst_day, day)
            # summary & figure
        if self.rmse_scores:
            print(f"\nAverage RMSE over {horizon_days} roll-outs: "
                f"{np.mean(self.rmse_scores):.4f}")
        plt.tight_layout(); plt.show()

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _plot_day(actual: pd.DataFrame, fcst: pd.DataFrame, day: pd.Timestamp):
        plt.plot(actual["ds"], actual["y"],  label=f"Actual {day.date()}",  lw=1.8)
        plt.plot(fcst["ds"],   fcst["yhat1"],label=f"Forecast {day.date()}", ls="--")

    def _update_rmse(self, act: pd.DataFrame, fcst: pd.DataFrame, day: pd.Timestamp):
        merged = (fcst.loc[:,["ds","yhat1"]]
                      .merge(act.loc[:,["ds","y"]], on="ds", how="inner"))
        if merged.empty:
            print(f"[{day.date()}] No overlap between forecast & actual.")
            return
        rmse = np.sqrt(((merged["y"] - merged["yhat1"])**2).mean())
        self.rmse_scores.append(rmse)
        print(f"[{day.date()}] RMSE = {rmse:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
#  RUN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pred = Predictor(
        csv_path="data_15min_measurements.csv",
        target_col="P+ Prejeta delovna moč",
        ts_col="Časovna značka"
    )
    pred.rolling_daily_forecast("01. 05. 2023 00:00:00", horizon_days=3)