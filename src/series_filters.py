from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesFilters:
    """Filtros opcionais na base de tickets (valores exatos como no ClickHouse)."""

    category: str | None = None
    ticket_type: str | None = None  # coluna `type` (ex.: Incidente / Requisição)
    assigned_team: str | None = None
