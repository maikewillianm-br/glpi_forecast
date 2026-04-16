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
