from __future__ import annotations

import pandas as pd


def complete_daily_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche dias sem eventos com y=0 (importante para séries temporais regulares).
    Espera colunas: day (datetime64[ns]), y (float).
    """
    if df.empty:
        return df

    out = df.sort_values("day").copy()
    full = pd.date_range(out["day"].min(), out["day"].max(), freq="D")
    base = pd.DataFrame({"day": full})
    merged = base.merge(out, on="day", how="left")
    merged["y"] = merged["y"].fillna(0.0)
    return merged
