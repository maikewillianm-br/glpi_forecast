from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ForecastResult:
    history: pd.Series
    fitted_in_sample: pd.Series
    forecast_index: pd.DatetimeIndex
    forecast_mean: np.ndarray
    forecast_ci_low: np.ndarray
    forecast_ci_high: np.ndarray
    order: tuple[int, int, int]
    aic: float | None


def time_series_split(
    series: pd.Series, train_ratio: float
) -> tuple[pd.Series, pd.Series]:
    n = len(series)
    cut = max(5, int(n * train_ratio))
    if cut >= n - 1:
        cut = n - 2
    train = series.iloc[:cut]
    test = series.iloc[cut:]
    return train, test


def fit_arima_forecast(
    series: pd.Series,
    order: tuple[int, int, int],
    horizon: int,
) -> ForecastResult:
    """Ajusta ARIMA na série completa e projeta `horizon` dias à frente."""
    model = ARIMA(series, order=order)
    fitted = model.fit()

    fc = fitted.get_forecast(steps=horizon)
    mean = fc.predicted_mean.to_numpy()
    conf = fc.conf_int(alpha=0.2)
    low = conf.iloc[:, 0].to_numpy()
    high = conf.iloc[:, 1].to_numpy()

    last_date = series.index.max()
    idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    return ForecastResult(
        history=series,
        fitted_in_sample=fitted.fittedvalues,
        forecast_index=idx,
        forecast_mean=mean,
        forecast_ci_low=low,
        forecast_ci_high=high,
        order=order,
        aic=float(fitted.aic) if fitted.aic is not None else None,
    )


def forecast_from_train(
    train: pd.Series, order: tuple[int, int, int], steps: int
) -> np.ndarray:
    """Previsão fora da amostra a partir apenas do treino (para métricas no teste)."""
    model = ARIMA(train, order=order)
    fitted = model.fit()
    return fitted.forecast(steps=steps).to_numpy()


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
