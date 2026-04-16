from __future__ import annotations

import pandas as pd

from .config import Settings


def get_client(settings: Settings):
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
    )


def fetch_daily_counts(settings: Settings) -> pd.DataFrame:
    """
    Agrega contagem diária a partir da tabela GLPI no ClickHouse.
    GLPI_DATE_FROM / GLPI_DATE_TO já validados em Settings (YYYY-MM-DD).
    """
    t = settings.glpi_events_table
    c = settings.glpi_date_column

    where_clauses: list[str] = ["1"]
    if settings.glpi_date_from:
        where_clauses.append(f"toDate(`{c}`) >= toDate('{settings.glpi_date_from}')")
    if settings.glpi_date_to:
        where_clauses.append(f"toDate(`{c}`) <= toDate('{settings.glpi_date_to}')")

    sql = f"""
    SELECT
        toDate(`{c}`) AS day,
        count() AS y
    FROM {t}
    WHERE {' AND '.join(where_clauses)}
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
