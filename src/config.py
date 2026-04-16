from __future__ import annotations

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

_ID = re.compile(r"^[\w]+$")
_QUAL = re.compile(r"^[\w]+\.[\w]+$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _optional_iso_date(label: str, value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    if not _ISO_DATE.match(v):
        raise ValueError(f"{label} deve estar no formato YYYY-MM-DD")
    return v


def _require_ident(name: str, value: str) -> str:
    if not value or not (_ID.match(value) or _QUAL.match(value)):
        raise ValueError(
            f"{name} inválido para SQL: use apenas letras, números, _ e opcionalmente db.tabela"
        )
    return value


@dataclass(frozen=True)
class Settings:
    clickhouse_host: str
    clickhouse_port: int
    clickhouse_user: str
    clickhouse_password: str
    clickhouse_database: str
    glpi_events_table: str
    glpi_date_column: str
    glpi_date_from: str | None
    glpi_date_to: str | None
    train_ratio: float
    arima_order: tuple[int, int, int]
    forecast_horizon: int


def load_settings() -> Settings:
    load_dotenv()

    host = os.getenv("CLICKHOUSE_HOST", "").strip()
    if not host:
        raise RuntimeError("CLICKHOUSE_HOST não definido no .env")

    port = int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123"))
    user = os.getenv("CLICKHOUSE_USER", "").strip()
    password = os.getenv("CLICKHOUSE_PASSWORD", "")
    database = os.getenv("CLICKHOUSE_DATABASE", "_master").strip()

    table = _require_ident(
        "GLPI_EVENTS_TABLE",
        os.getenv("GLPI_EVENTS_TABLE", "glpi_tickets").strip(),
    )
    date_col = _require_ident(
        "GLPI_DATE_COLUMN",
        os.getenv("GLPI_DATE_COLUMN", "date").strip(),
    )

    df_raw = _optional_iso_date("GLPI_DATE_FROM", os.getenv("GLPI_DATE_FROM"))
    dt_raw = _optional_iso_date("GLPI_DATE_TO", os.getenv("GLPI_DATE_TO"))

    train_ratio = float(os.getenv("TRAIN_RATIO", "0.75"))
    if not 0.5 < train_ratio < 1:
        raise ValueError("TRAIN_RATIO deve estar entre 0.5 e 1")

    order_parts = os.getenv("ARIMA_ORDER", "1,1,1").split(",")
    if len(order_parts) != 3:
        raise ValueError("ARIMA_ORDER deve ser p,d,q separados por vírgula, ex: 1,1,1")
    arima_order = tuple(int(x.strip()) for x in order_parts)  # type: ignore[assignment]

    horizon = int(os.getenv("FORECAST_HORIZON", "14"))
    if horizon < 1:
        raise ValueError("FORECAST_HORIZON deve ser >= 1")

    return Settings(
        clickhouse_host=host,
        clickhouse_port=port,
        clickhouse_user=user,
        clickhouse_password=password,
        clickhouse_database=database,
        glpi_events_table=table,
        glpi_date_column=date_col,
        glpi_date_from=df_raw,
        glpi_date_to=dt_raw,
        train_ratio=train_ratio,
        arima_order=arima_order,  # type: ignore[arg-type]
        forecast_horizon=horizon,
    )
