"""
Pipeline: ClickHouse (GLPI) -> série diária -> ARIMA -> gráfico e métricas.

Uso (na pasta do projeto):
    python -m scripts.run_forecast
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src.clickhouse_io import fetch_daily_counts
from src.config import load_settings
from src.forecast_model import (
    fit_arima_forecast,
    forecast_from_train,
    mae,
    time_series_split,
)
from src.prepare_series import complete_daily_calendar


def main() -> None:
    settings = load_settings()
    raw = fetch_daily_counts(settings)
    if raw.empty:
        raise SystemExit(
            "Nenhuma linha retornada do ClickHouse. Verifique tabela, coluna de data e filtros."
        )

    daily = complete_daily_calendar(raw)
    series = daily.set_index("day")["y"].sort_index().asfreq("D")

    train, test = time_series_split(series, settings.train_ratio)
    if len(test) > 0:
        pred_test = forecast_from_train(
            train, settings.arima_order, steps=len(test)
        )
        print(f"MAE no conjunto de teste ({len(test)} dias): {mae(test.to_numpy(), pred_test):.4f}")

    result = fit_arima_forecast(
        series, settings.arima_order, settings.forecast_horizon
    )
    if result.aic is not None:
        print(f"AIC (modelo na série completa): {result.aic:.2f}")
    print(f"Ordem ARIMA: {result.order}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    result.history.plot(ax=ax, label="Histórico (y)", color="C0")
    result.fitted_in_sample.plot(ax=ax, label="Ajuste in-sample", color="C1", alpha=0.7)
    ax.plot(
        result.forecast_index,
        result.forecast_mean,
        label="Previsão",
        color="C2",
    )
    ax.fill_between(
        result.forecast_index,
        result.forecast_ci_low,
        result.forecast_ci_high,
        color="C2",
        alpha=0.2,
        label="IC 80%",
    )
    ax.set_title("Série diária GLPI (ClickHouse) — ARIMA")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Contagem")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = out_dir / "forecast_arima.png"
    fig.savefig(path, dpi=120)
    print(f"Figura salva em: {path}")

    cache = ROOT / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache / "series_daily.csv", index=False)
    print(f"Série diária em cache: {cache / 'series_daily.csv'}")


if __name__ == "__main__":
    main()
