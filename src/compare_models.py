"""
Ajuste e previsão multi-step para comparação de modelos (notebook + Shiny).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = float(np.mean(np.abs(y_true - y_pred)))
    r = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return m, r


def fit_predict_arima_on_test_index(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    h = len(test_index)
    m = ARIMA(train, order=(1, 1, 1)).fit()
    return pd.Series(m.forecast(steps=h).to_numpy(), index=test_index, name="yhat")


def fit_predict_sarima(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    h = len(test_index)
    if len(train) < 21:
        raise ValueError("treino < 21 dias")
    m = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return pd.Series(m.forecast(steps=h).to_numpy(), index=test_index, name="yhat")


def fit_predict_holt_winters(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    h = len(test_index)
    if len(train) >= 21:
        m = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        ).fit(optimized=True)
        pred = m.forecast(h).to_numpy()
    else:
        m = ExponentialSmoothing(train, trend="add", seasonal=None).fit(optimized=True)
        pred = m.forecast(h).to_numpy()
    return pd.Series(pred, index=test_index, name="yhat")


def fit_predict_prophet(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """
    Prophet com sazonalidades adicionais:
    - semanal (mais harmónicos), anual se houver ≥ ~1 ano de treino,
    - mensal (~30.5d) quando há histórico suficiente,
    - `seasonality_prior_scale` / `changepoint_prior_scale` afinados para counts diários.
    """
    from prophet import Prophet

    h = len(test_index)
    ds = pd.to_datetime(train.index)
    if getattr(ds, "tz", None) is not None:
        ds = ds.tz_convert("UTC").tz_localize(None)
    df_tr = pd.DataFrame({"ds": ds, "y": train.astype(float).values})
    n = len(df_tr)

    yearly_fourier = 10 if n >= 366 else 0
    weekly_fourier = 10

    m_pr = Prophet(
        daily_seasonality=False,
        weekly_seasonality=weekly_fourier,
        yearly_seasonality=yearly_fourier if yearly_fourier else False,
        seasonality_mode="additive",
        seasonality_prior_scale=12.0,
        changepoint_prior_scale=0.08,
    )

    if n >= 90:
        m_pr.add_seasonality(
            name="monthly",
            period=30.5,
            fourier_order=5,
            prior_scale=8.0,
        )

    m_pr.fit(df_tr)
    fut = m_pr.make_future_dataframe(periods=h, freq="D", include_history=False)
    fc = m_pr.predict(fut)
    fc["ds"] = pd.to_datetime(fc["ds"]).dt.normalize()
    yhat_s = fc.set_index("ds")["yhat"]
    tidx = pd.to_datetime(test_index).normalize()
    yhat = yhat_s.reindex(tidx).interpolate().bfill().ffill().to_numpy()
    return pd.Series(np.maximum(yhat, 0.0), index=test_index, name="yhat")


def fit_predict_xgboost_recursive(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    import xgboost as xgb

    h = len(test_index)
    nlags = min(14, max(3, len(train) // 10))

    def build_supervised(y: np.ndarray, nlags_: int) -> tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        for i in range(nlags_, len(y)):
            X.append(y[i - nlags_ : i])
            Y.append(y[i])
        return np.asarray(X), np.asarray(Y)

    hist = list(train.astype(float).values)
    tr_y = np.asarray(hist, dtype=float)
    X_tr, y_tr = build_supervised(tr_y, nlags)
    if len(y_tr) < 10:
        raise ValueError("poucos pontos para XGBoost")
    xgb_m = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        verbosity=0,
    )
    xgb_m.fit(X_tr, y_tr)
    preds = []
    for _ in range(h):
        x_row = np.asarray(hist[-nlags:], dtype=float).reshape(1, -1)
        p = float(xgb_m.predict(x_row)[0])
        p = max(0.0, p)
        preds.append(p)
        hist.append(p)
    return pd.Series(np.asarray(preds), index=test_index, name="yhat")


def fit_one_model(model_key: str, train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    if model_key == "arima":
        return fit_predict_arima_on_test_index(train, test_index)
    if model_key == "sarima":
        return fit_predict_sarima(train, test_index)
    if model_key == "hw":
        return fit_predict_holt_winters(train, test_index)
    if model_key == "prophet":
        return fit_predict_prophet(train, test_index)
    if model_key == "xgboost":
        return fit_predict_xgboost_recursive(train, test_index)
    raise ValueError(f"modelo desconhecido: {model_key}")


def run_all_models(train: pd.Series, test: pd.Series) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Executa todos os modelos; devolve (previsões_por_nome, erros_por_nome)."""
    predictions: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    test_index = test.index
    runners = [
        ("ARIMA(1,1,1)", "arima"),
        ("SARIMA(1,1,1)(1,0,1,7)", "sarima"),
        ("Holt–Winters (s=7)", "hw"),
        ("Prophet (sazonalidades estendidas)", "prophet"),
        ("XGBoost (lags recursivos)", "xgboost"),
    ]
    for label, key in runners:
        try:
            predictions[label] = fit_one_model(key, train, test_index)
        except Exception as e:
            errors[label] = str(e)
    return predictions, errors
