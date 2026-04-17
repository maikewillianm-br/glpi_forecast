# GLPI + ClickHouse — modelagem de séries temporais (forecast)

Projeto acadêmico: extrair uma série diária (ex.: chamados/dia) de tabelas estilo GLPI materializadas no **ClickHouse**, ajustar um **ARIMA** com `statsmodels` e gerar previsão com gráfico.

## Pré-requisitos

- Python 3.10+
- Acesso HTTP ao ClickHouse (porta padrão **8123**)
- Tabelas `dw.glpi_tickets` e `dw.vw_glpi_tickets` acessíveis com a query padrão do projeto

## Configuração

1. Crie o ambiente virtual e instale dependências:

```text
cd glpi_clickhouse_forecast
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Copie `.env.example` para `.env` e preencha com suas credenciais do ClickHouse.

3. Ajuste no `.env`, se necessário:

- `GLPI_DATE_FROM` / `GLPI_DATE_TO` — opcional, formato `YYYY-MM-DD` (filtro em `dt_created`; se `GLPI_DATE_FROM` estiver vazio, usa `2023-01-01`)
- `ARIMA_ORDER` — ordem `p,d,q` (ex.: `1,1,1`)
- `TRAIN_RATIO` — proporção para treino na métrica de teste
- `FORECAST_HORIZON` — dias à frente na figura final

## Execução

Na pasta do projeto (com `.venv` ativado):

```text
python scripts/run_forecast.py
```

### Shiny (filtros + modelos)

Interface web ([Shiny for Python](https://shiny.posit.co/py/)) para escolher **categoria**, **tipo** (`Incidente` / `Requisição`, etc.) e **equipa**, ajustar o tamanho da janela de teste e ver **MAE/RMSE** e gráficos para o modelo selecionado (ou comparar todos).

```text
pip install shiny
cd glpi_clickhouse_forecast
shiny run shiny_app/app.py --reload
```

Abre o URL indicado no terminal (por defeito `http://127.0.0.1:8000`). Com filtros ativos é necessário **ClickHouse**; sem filtros, se a ligação falhar, usa-se `data/cache/series_daily.csv` como no notebook.

Saídas:

- `outputs/forecast_arima.png` — histórico, ajuste e previsão com intervalo de confiança aproximado
- `data/cache/series_daily.csv` — série diária usada no modelo

## Consulta SQL usada

A série diária é `count()` por `toDate(dt_created)` sobre o conjunto filtrado abaixo (equivalente à query analítica em `dw`):

```sql
SELECT
    toDate(dt_created) AS day,
    count() AS y
FROM (
    SELECT id_key, title, dt_created, /* ... demais colunas ... */
    FROM dw.glpi_tickets AS t
    WHERE t._process_date_brt = (SELECT max(_process_date_brt) FROM dw.vw_glpi_tickets)
      AND toDate(t.dt_created) >= toDate('<GLPI_DATE_FROM ou 2023-01-01>')
      /* opcional: AND toDate(t.dt_created) <= toDate('<GLPI_DATE_TO>') */
      AND upperUTF8(t.title) NOT LIKE '%TESTE%'
) AS tickets
GROUP BY day
ORDER BY day
```

Detalhe da subconsulta: `src/clickhouse_io.py` (`_sql_fetch_tickets_base`).

## Notas para o trabalho

- Documente a escolha da **ordem ARIMA** (ou experimente grades simples e compare AIC / MAE).
- Comente limitações: sazonalidade semanal forte pode exigir **SARIMA** ou modelos com regressores.
- Não versione o arquivo `.env` (já está no `.gitignore`).

## Publicar no GitHub

O repositório Git local já está inicializado na pasta do projeto (branch `main`, commit inicial).

### Opção A — GitHub CLI (`gh`, recomendado)

Se o PowerShell disser que `gh` não é reconhecido, o PATH ainda não foi atualizado nessa janela. **Feche e abra o terminal de novo**, ou execute:

```text
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

Alternativa: use o caminho completo do executável:

```text
& "C:\Program Files\GitHub CLI\gh.exe" auth login
```

1. Faça login (abre o navegador ou pede token):

```text
gh auth login
```

2. Crie o repositório remoto e envie o código:

```text
cd glpi_clickhouse_forecast
gh repo create glpi_clickhouse_forecast --public --source=. --remote=origin --push
```

Se o nome `glpi_clickhouse_forecast` já existir na sua conta, use outro nome no comando.

### Opção B — site do GitHub + `git push`

1. Em [github.com/new](https://github.com/new), crie um repositório **vazio** (sem README).
2. Na pasta do projeto:

```text
cd glpi_clickhouse_forecast
git remote add origin https://github.com/SEU_USUARIO/NOME_DO_REPO.git
git push -u origin main
```

Substitua `SEU_USUARIO` e `NOME_DO_REPO` pelos valores reais. Use autenticação por **Personal Access Token** (HTTPS) ou SSH, conforme sua configuração.
