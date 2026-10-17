# CommerceOps AI Engine

[![CI](https://github.com/boucoding/CommerceOps-AI-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/boucoding/CommerceOps-AI-Engine/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)
![PostgreSQL 16 + pgvector](https://img.shields.io/badge/PostgreSQL%2016-pgvector-336791.svg)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

An LLM agent that reads customer emails for an online store, works out what each customer wants,
checks the catalog and sales data, reserves stock, and writes the reply. It uses the
**plan-and-execute** pattern, with a **sandboxed SQL tool** for research.

Illustrative run:

```
"Hi, this is Liz. Please send me 5 CLF2109 Cable Knit Beanies and 2 pairs of FZZ1098 Fuzzy Slippers."

plan     order request · search_catalog → search_catalog → reserve_stock → submit_reply
execute  search_catalog("CLF2109")   → Cable Knit Beanie, 2 in stock
         search_catalog("FZZ1098")   → Fuzzy Slippers, 2 in stock
         reserve_stock(...)          → CLF2109 ×5 out of stock · FZZ1098 ×2 created
         submit_reply(...)           → validated, signed, sent
```

---

## How it works

```mermaid
flowchart LR
    E[Email] --> G{Guardrail}
    G -- injection --> H[Escalate to a person]
    G -- PII masked --> P[Planner]
    P --> X[Executor<br/>one tool per turn]
    X -- tool error --> R[Replanner]
    R -- continue / revise --> X
    R -- escalate --> H
    X --> T[(Tools)]
    T --> X
    X -- submit_reply passes validation --> O[Reply + order lines + trace]
```

1. **Guardrail.** Emails matching prompt-injection signatures go straight to a person; the model never
   sees them. Emails, phone numbers, card numbers and addresses are replaced with tokens such as
   `{{EMAIL_1}}` before anything reaches the model, and restored only in the outgoing reply.
2. **Plan.** One structured-output call returns the intent, the customer's name and language, and a
   short list of tool steps with success criteria.
3. **Execute.** A strict function-calling loop works through the plan, one tool per turn:

   | Tool | What it does |
   |---|---|
   | `search_catalog` | Hybrid search: exact or misspelled product IDs (pg_trgm), plus dense vectors (pgvector HNSW) and full-text, fused with RRF |
   | `get_products` | Live details and stock |
   | `query_database` | Read-only SQL over two views: `product_catalog` and `order_history` |
   | `reserve_stock` | Atomic, all-or-nothing per line, idempotent per email. Only offered for orders |
   | `submit_reply` | Ends the run, but only if the reply passes validation |

4. **Replan.** When a tool fails, a replanner sees the goal, the plan and the history. It decides to
   *continue*, *revise* the remaining steps, or *escalate* to a person.
5. **Finish.** The outcome is stored with the plan, the full tool trace and the token usage. Every email
   ends in one of these statuses:
   - `processed`
   - `needs_clarification`
   - `escalated`
   - `needs_review` (the agent gave up or ran out of budget)
   - `failed` (model or infrastructure error; safe to retry)

---

## Safety by design

| Risk | Control |
|---|---|
| Prompt injection | Signature screen before any model call; customer text sits inside `<customer_untrusted_input>` tags; tools enforce policy in code, so the model cannot "talk its way" past them |
| Leaking personal data to the model | Reversible PII masking; the SQL views expose no customer data; logs never contain email content |
| Model-written SQL | sqlglot validation fails closed: one `SELECT`, allowlisted views and functions, no DML/DDL/`SELECT INTO`/locks. Then a `READ ONLY` transaction with `SET LOCAL ROLE` to a NOLOGIN role that can only read the two views. Also: `statement_timeout`, caps on rows and cell length, and an audit log of every attempt |
| Wrong or invented products | `reserve_stock` only accepts products a tool returned in this run. Replies may only cite product IDs that tools returned, and must mention every reserved product |
| Overselling and double charging | Row locks taken in primary-key order; all-or-nothing lines; reservations unique per (email, product), so retries never reserve twice |
| Runaway cost or loops | Caps on turns and replans per email, one tool call per turn, quantity limits that hand off to a person |

Tests prove that the database rejects writes and private tables even when the SQL parser is bypassed.

---

## Quick start

**Docker** (PostgreSQL + API):

```bash
cp .env.example .env     # set OPENAI_API_KEY and POSTGRES_PASSWORD
docker compose up -d --build
curl -s localhost:8000/v1/emails -H 'content-type: application/json' \
  -d '{"email_id": "demo-1", "body": "Do you have warm winter hats? Which one sells best?"}'
```

Interactive API docs: http://localhost:8000/docs

**Batch** (CSV in → Excel/CSV report out):

```bash
docker compose --profile batch run --rm batch         # writes ./exports
```

**Local development** (Python 3.12, any PostgreSQL 16 with pgvector; connection settings in `.env`):

```bash
make dev                  # install dependencies
make run                  # process data/inbound_customer_emails.csv
make serve                # API on :8000
make help                 # all commands
```

Database migrations run automatically on startup, and the catalog in `data/catalog_products.csv` is
synced, re-embedding only products whose text changed. Existing stock levels are kept.

---

## Evaluation

[`evals/gold_labels.json`](evals/gold_labels.json) holds hand labels for the 23 sample emails:
- intent
- exact order lines (product, quantity, status)
- status
- customer name
- reply language
- products the reply must be grounded in

The evaluation scores intent accuracy, order-line precision and recall, status, name and language
accuracy, context recall, reply grounding, and **plan adherence** (did the executor carry out the steps it
planned?). It also counts tool calls, replans, SQL queries and tokens.

```bash
make record   # run the agent live on the sample emails, save every model exchange (needs a key)
make eval     # replay that run deterministically and score it (no key, no network)
```

Both commands **reset the database** first (processed emails are cleared and stock is restored from the
catalog file), so point them at a development database.

Recordings are strict: a changed prompt, tool schema or catalog row makes the replay fail rather than
return an answer to a different question. Once `fixtures/llm_recordings.json` is committed, CI re-scores
it on every push and fails if intent accuracy or order-line recall drops below 90%.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/emails` | Process one email; idempotent on `email_id`. Returns status, reply, order lines, plan and trace |
| `POST` | `/v1/emails/batch` | Process up to `API_MAX_BATCH` emails |
| `GET` | `/v1/emails/{email_id}` | Stored outcome |
| `GET` | `/v1/products/{product_id}` | Product with live stock |
| `GET` | `/v1/products?query=…` | The same hybrid search the agent uses |
| `GET` | `/healthz`, `/readyz` | Liveness and readiness |

Set `API_KEY` to require an `X-API-Key` header on `/v1` routes. Every response carries `X-Request-ID`,
which also appears in the logs.

---

## Configuration

All settings come from environment variables or `.env`; see [.env.example](.env.example).

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODE` | `live` | `live`, `record` or `replay` |
| `OPENAI_API_KEY` | — | Required for `live` and `record` |
| `LLM_MODEL` | `gpt-6-luna` | Needs Structured Outputs and function calling |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 1536 dimensions (matches the DB column) |
| `AGENT_MAX_STEPS` / `AGENT_MAX_REPLANS` | `12` / `2` | Per-email budgets |
| `AGENT_MAX_ORDER_QUANTITY` | `50` | Larger requests go to human review |
| `SQL_STATEMENT_TIMEOUT_MS` / `SQL_MAX_ROWS` | `2000` / `50` | Sandbox limits |
| `BATCH_WORKERS` | `1` | `1` = first-come-first-served stock allocation; more is safe but allocation order varies |
| `DATABASE_URL` or `POSTGRES_*` | — | Database connection |
| `BRAND_NAME`, `AGENT_NAME`, `SUPPORT_EMAIL`, `PHONE_NUMBER` | Aura Fashion Atelier… | Prompts and signature |

---

## Development

```bash
export TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/commerceops_test   # wiped by the tests
make check                # ruff + mypy + pytest with coverage
make docker-test          # the same suite inside the Docker stack
```

- **94 tests, 94% coverage.** Tests that don't need a database (guardrails, SQL validation, reply
  validation, LLM clients, scoring) run anywhere; the rest skip without `TEST_DATABASE_URL`.
- **No real model in tests.** A scripted client plays back plans and tool calls. Tests cover replanning,
  recovery from rejected replies, outages followed by a retry, crash recovery without double
  reservation, budget exhaustion, injection escalation and PII never reaching the model.
- **CI** runs lint, types and tests against a pgvector service, replays the recorded evaluation, and runs
  the suite inside the Docker stack.

```
src/
  agent/          planner, executor loop, tools, prompts, schemas
  sandbox/        SQL validation and read-only execution
  safety/         input guardrails (injection, PII) and reply validation
  llm/            OpenAI client, record/replay
  storage/        PostgreSQL store, schema migration, connection pool
  api/            FastAPI app
  models/         domain objects (products, emails, order lines, outcomes)
  utils/          logging, Excel/CSV reporting
  engine.py       wiring shared by the CLI, API and evaluation
  catalog.py      catalog sync (embeds only changed products)
  evaluation.py   gold-label scoring
  cli.py          run | eval | migrate | seed | serve
data/             sample catalog (99 products) and inbound emails (23)
evals/            gold labels
fixtures/         recorded model traffic (created by `make record`)
docker/           database init script (creates the test database)
```

---

## Limitations

- Needs a model API key to process new emails. Replay only covers emails that were recorded.
- The injection screen is signature-based; the main protection is that tools enforce policy in code.
- One tool call per turn trades latency for simpler permission checks and traces.
- No mailbox connector (IMAP, helpdesk webhooks). The HTTP API is the integration point.

---

Created by [boucoding](https://github.com/boucoding) · [LinkedIn](https://www.linkedin.com/in/aadiab) ·
Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): free for personal,
educational and research use; commercial use needs the author's written consent.
