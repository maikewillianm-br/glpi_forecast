"""
Shiny for Python — explorar série GLPI filtrada e comparar modelos de previsão.

Documentação: https://shiny.posit.co/py/

Na raiz do projeto:
    pip install shiny
    shiny run shiny_app/app.py --reload
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, reactive, render, ui

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clickhouse_io import fetch_daily_counts, fetch_filter_options
from src.compare_models import fit_one_model, mae_rmse, run_all_models
from src.config import load_settings
from src.prepare_series import complete_daily_calendar
from src.series_filters import SeriesFilters

_CACHE_CSV = ROOT / "data" / "cache" / "series_daily.csv"

MODEL_CHOICES = {
    "sarima": "SARIMA (1,1,1)x(1,0,1,7)",
    "hw": "Holt–Winters (s=7)",
    "prophet": "Prophet (sazonalidades estendidas)",
    "xgboost": "XGBoost (lags recursivos)",
    "arima": "ARIMA(1,1,1)",
}


def _empty_filter_choices() -> dict[str, str]:
    return {"": "(todas)"}


def _build_select_choices(values: list[str]) -> dict[str, str]:
    d = _empty_filter_choices()
    for v in values:
        d[v] = v
    return d


def _load_filter_choices() -> tuple[dict[str, str], dict[str, str], dict[str, str], str | None]:
    err = None
    try:
        settings = load_settings()
        opts = fetch_filter_options(settings)
        return (
            _build_select_choices(opts.get("category", [])),
            _build_select_choices(opts.get("ticket_type", [])),
            _build_select_choices(opts.get("assigned_team", [])),
            None,
        )
    except Exception as e:
        err = str(e)
        e0 = _empty_filter_choices()
        return e0, e0, e0, err


_CAT_CHOICES, _TYPE_CHOICES, _TEAM_CHOICES, _BOOT_ERR = _load_filter_choices()

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Filtros (ClickHouse)"),
        ui.input_select("category", "Categoria / área", choices=_CAT_CHOICES, selected=""),
        ui.input_select("ticket_type", "Tipo (Incidente, Requisição, …)", choices=_TYPE_CHOICES, selected=""),
        ui.input_select("assigned_team", "Equipa atribuída", choices=_TEAM_CHOICES, selected=""),
        ui.hr(),
        ui.input_slider("test_h", "Dias na janela de teste", min=7, max=45, value=28),
        ui.input_select("model", "Modelo a visualizar", choices=MODEL_CHOICES, selected="sarima"),
        ui.input_action_button("run_all", "Comparar todos os modelos (MAE/RMSE)"),
        ui.hr(),
        ui.p(
            "Base: último snapshot em ",
            ui.code("dw.vw_glpi_tickets"),
            ", agregação diária por ",
            ui.code("dt_created"),
            ".",
            class_="small text-muted",
        ),
        width=340,
    ),
    ui.page_fillable(
        ui.h2("Forecast GLPI — Shiny"),
        ui.output_text_verbatim("boot_status"),
        ui.layout_columns(
            ui.card(ui.card_header("Métricas (teste)"), ui.output_text_verbatim("metrics_text")),
            ui.card(ui.card_header("Ranking (última comparação)"), ui.output_table("ranking_table")),
            col_widths=(6, 6),
        ),
        ui.card(ui.card_header("Real vs previsto (teste)"), ui.output_plot("main_plot", height="420px")),
        ui.card(ui.card_header("Histórico recente da série filtrada"), ui.output_plot("history_plot", height="280px")),
        padding=16,
    ),
)


def server(input, output, _session):
    ranking_state = reactive.Value(pd.DataFrame())

    @reactive.calc
    def boot_message() -> str:
        if _BOOT_ERR:
            return f"Aviso ao carregar opções de filtro (usa só '(todas)'): {_BOOT_ERR}"
        return ""

    @output
    @render.text
    def boot_status():
        return boot_message()

    @reactive.calc
    def filters_obj() -> SeriesFilters | None:
        c = input.category() or None
        t = input.ticket_type() or None
        tm = input.assigned_team() or None
        if not c and not t and not tm:
            return None
        return SeriesFilters(category=c, ticket_type=t, assigned_team=tm)

    @reactive.calc
    def daily_series() -> pd.DataFrame:
        fl = filters_obj()
        try:
            settings = load_settings()
            raw = fetch_daily_counts(settings, fl)
            if raw.empty:
                raise ValueError("Série vazia para estes filtros")
            return complete_daily_calendar(raw)
        except Exception as e:
            if fl is not None:
                raise RuntimeError(
                    "Não foi possível obter dados filtrados do ClickHouse. "
                    f"Erro: {e}"
                ) from e
            if not _CACHE_CSV.exists():
                raise
            daily = pd.read_csv(_CACHE_CSV, parse_dates=["day"])
            return complete_daily_calendar(daily)

    @reactive.calc
    def train_test_split():
        daily = daily_series()
        s = daily.set_index("day")["y"].sort_index().asfreq("D")
        h = int(input.test_h())
        h = min(h, max(7, len(s) - 10))
        train = s.iloc[:-h].copy()
        test = s.iloc[-h:].copy()
        return train, test

    @output
    @render.text
    def metrics_text():
        train, test = train_test_split()
        key = input.model()
        try:
            yhat = fit_one_model(key, train, test.index)
            m, r = mae_rmse(test.values, yhat.reindex(test.index).values)
            return f"Modelo: {MODEL_CHOICES[key]}\nTreino: {len(train)} dias | Teste: {len(test)} dias\nMAE:  {m:.4f}\nRMSE: {r:.4f}"
        except Exception as e:
            return f"Erro ao ajustar modelo: {e}"

    @output
    @render.plot
    def main_plot():
        train, test = train_test_split()
        key = input.model()
        fig, ax = plt.subplots(figsize=(10, 4))
        try:
            yhat = fit_one_model(key, train, test.index)
            ax.plot(test.index, test.values, "k-", label="Real (teste)", linewidth=2)
            ax.plot(yhat.index, yhat.reindex(test.index).values, "--", label=MODEL_CHOICES[key], alpha=0.9)
            ax.legend(loc="upper left", fontsize=8)
            ax.set_title("Janela de teste")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Dia")
        ax.set_ylabel("Contagem")
        fig.tight_layout()
        return fig

    @output
    @render.plot
    def history_plot():
        train, test = train_test_split()
        s = pd.concat([train, test])
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(s.index, s.values, color="steelblue", linewidth=0.8)
        ax.axvline(test.index.min(), color="red", linestyle=":", alpha=0.8, label="Início teste")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Série diária (filtros atuais)")
        fig.tight_layout()
        return fig

    @reactive.effect
    @reactive.event(input.run_all)
    def _do_ranking():
        train, test = train_test_split()
        preds, errs = run_all_models(train, test)
        rows = []
        for name, yhat in preds.items():
            yhat = yhat.reindex(test.index)
            m, r = mae_rmse(test.values, yhat.values)
            rows.append({"modelo": name, "MAE": m, "RMSE": r})
        df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
        if errs:
            print("Modelos com erro:", errs)
        ranking_state.set(df)

    @output
    @render.table
    def ranking_table():
        df = ranking_state.get()
        if df.empty:
            return pd.DataFrame({"": ["Clique em «Comparar todos os modelos»"]})
        return df


app = App(app_ui, server)
