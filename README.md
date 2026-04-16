# GLPI + ClickHouse — modelagem de séries temporais (forecast)

Projeto acadêmico: extrair uma série diária (ex.: chamados/dia) de tabelas estilo GLPI materializadas no **ClickHouse**, ajustar um **ARIMA** com `statsmodels` e gerar previsão com gráfico.

## Pré-requisitos

- Python 3.10+
- Acesso HTTP ao ClickHouse (porta padrão **8123**)
- Tabela com coluna de data compatível com `toDate()` no ClickHouse

## Configuração

1. Crie o ambiente virtual e instale dependências:

```text
cd glpi_clickhouse_forecast
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Copie `.env.example` para `.env` e preencha com suas credenciais e nomes de tabela/coluna reais no ClickHouse.

3. Ajuste no `.env`, se necessário:

- `GLPI_EVENTS_TABLE` — tabela de eventos (ex.: `glpi_tickets` ou `meu_db.glpi_tickets`)
- `GLPI_DATE_COLUMN` — coluna da data de abertura ou ocorrência (ex.: `date`)
- `GLPI_DATE_FROM` / `GLPI_DATE_TO` — opcional, formato `YYYY-MM-DD`
- `ARIMA_ORDER` — ordem `p,d,q` (ex.: `1,1,1`)
- `TRAIN_RATIO` — proporção para treino na métrica de teste
- `FORECAST_HORIZON` — dias à frente na figura final

## Execução

Na pasta do projeto (com `.venv` ativado):

```text
python scripts/run_forecast.py
```

Saídas:

- `outputs/forecast_arima.png` — histórico, ajuste e previsão com intervalo de confiança aproximado
- `data/cache/series_daily.csv` — série diária usada no modelo

## Consulta SQL usada

A agregação é equivalente a:

```sql
SELECT toDate(`<GLPI_DATE_COLUMN>`) AS day, count() AS y
FROM <GLPI_EVENTS_TABLE>
WHERE ... filtros de data opcionais ...
GROUP BY day ORDER BY day
```

Se o seu schema usar outro nome de tabela, view ou coluna de tempo, altere apenas o `.env`.

## Notas para o trabalho

- Documente a escolha da **ordem ARIMA** (ou experimente grades simples e compare AIC / MAE).
- Comente limitações: sazonalidade semanal forte pode exigir **SARIMA** ou modelos com regressores.
- Não versione o arquivo `.env` (já está no `.gitignore`).
