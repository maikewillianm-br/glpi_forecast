from __future__ import annotations

import pandas as pd

from .config import Settings
from .series_filters import SeriesFilters

# Limite inferior padrão de dt_created (alinhado à query analítica original).
_DEFAULT_CREATED_FROM = "2023-01-01"


def get_client(settings: Settings):
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
    )


def _ch_literal(value: str) -> str:
    """Literal SQL string seguro (escape de aspas)."""
    s = str(value).replace("\\", "\\\\").replace("'", "''")
    return f"'{s}'"


def _sql_fetch_tickets_base(settings: Settings, filters: SeriesFilters | None = None) -> str:
    """
    Base fixa: dw.glpi_tickets + último snapshot em vw_glpi_tickets,
    filtros de criação e exclusão de títulos de teste.
    Filtros opcionais: categoria (área), tipo GLPI, equipa atribuída.
    """
    date_from = settings.glpi_date_from or _DEFAULT_CREATED_FROM
    extra_to = ""
    if settings.glpi_date_to:
        extra_to = f"        AND toDate(t.dt_created) <= toDate('{settings.glpi_date_to}')\n"

    extra_dim = ""
    if filters is not None:
        if filters.category:
            extra_dim += f"        AND t.category = {_ch_literal(filters.category)}\n"
        if filters.ticket_type:
            extra_dim += f"        AND t.`type` = {_ch_literal(filters.ticket_type)}\n"
        if filters.assigned_team:
            extra_dim += f"        AND t.assigned_team = {_ch_literal(filters.assigned_team)}\n"

    return f"""
    SELECT
        id_key,
        title,
        dt_created,
        dt_closed,
        dt_solved,
        dt_resolution_deadline,
        hours_resolution_planned,
        hours_resolution,
        status,
        urgency,
        priority,
        nm_requester,
        reporter,
        category,
        `type`,
        assigned_user,
        assigned_team,
        id_assigned_team,
        id_assigned_user,
        id_reported_user,
        sla,
        _date,
        _process_date,
        _process_date_brt
    FROM dw.glpi_tickets AS t
    WHERE
        t._process_date_brt = (
            SELECT max(_process_date_brt)
            FROM dw.vw_glpi_tickets
        )
        AND toDate(t.dt_created) >= toDate('{date_from}')
{extra_to}{extra_dim}        AND upperUTF8(t.title) NOT LIKE '%TESTE%'
"""


def fetch_daily_counts(
    settings: Settings,
    filters: SeriesFilters | None = None,
) -> pd.DataFrame:
    """
    Contagem diária de tickets (y) por dia de criação (dt_created),
    a partir da base analítica em dw.glpi_tickets.
    """
    inner = _sql_fetch_tickets_base(settings, filters).strip()
    sql = f"""
    SELECT
        toDate(dt_created) AS day,
        count() AS y
    FROM (
{inner}
    ) AS tickets
    GROUP BY day
    ORDER BY day
    """

    client = get_client(settings)
    df = client.query_df(sql)
    if df.empty:
        return df

    df["day"] = pd.to_datetime(df["day"])
    df["y"] = df["y"].astype(float)
    return df


def fetch_filter_options(settings: Settings) -> dict[str, list[str]]:
    """
    Valores distintos de categoria, tipo e equipa no mesmo snapshot da query base
    (para preencher filtros na UI).
    """
    date_from = settings.glpi_date_from or _DEFAULT_CREATED_FROM
    extra_to = ""
    if settings.glpi_date_to:
        extra_to = f"      AND toDate(t.dt_created) <= toDate('{settings.glpi_date_to}')\n"

    sql = f"""
    SELECT DISTINCT
        trim(toString(t.category)) AS category,
        trim(toString(t.`type`)) AS ticket_type,
        trim(toString(t.assigned_team)) AS assigned_team
    FROM dw.glpi_tickets AS t
    WHERE
        t._process_date_brt = (SELECT max(_process_date_brt) FROM dw.vw_glpi_tickets)
        AND toDate(t.dt_created) >= toDate('{date_from}')
{extra_to}      AND upperUTF8(t.title) NOT LIKE '%TESTE%'
    """
    client = get_client(settings)
    df = client.query_df(sql)
    out: dict[str, list[str]] = {"category": [], "ticket_type": [], "assigned_team": []}
    if df.empty:
        return out

    def uniq(col: str) -> list[str]:
        s = df[col].dropna().astype(str).str.strip()
        s = s[s != ""]
        return sorted(s.unique().tolist())

    out["category"] = uniq("category")
    out["ticket_type"] = uniq("ticket_type")
    out["assigned_team"] = uniq("assigned_team")
    return out
