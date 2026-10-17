# CommerceOps AI Engine
### LLM-assisted order fulfilment and customer-inquiry automation for e-commerce email

[![CI](https://github.com/boucoding/CommerceOps-AI-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/boucoding/CommerceOps-AI-Engine/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)
![PostgreSQL 16 + pgvector](https://img.shields.io/badge/PostgreSQL%2016-pgvector%20HNSW-336791.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Support teams at fashion and lifestyle retailers spend minutes per email working out what the customer wants,
checking stock and writing the reply. CommerceOps turns each inbound email into structured actions:

1. **Classifies** it as an *order request* or a *product inquiry* (any language).
2. For orders, **identifies the products and quantities**, even from typos (`DHN0987` → `CHN0987`), spaced
   SKUs (`[CBT 89 01]`), names ("retro sun glasses") and phrases like "all the remaining" or "three to four",
   then **reserves stock atomically**.
3. **Drafts a reply** grounded only in verified catalog facts and the reservation result, with suggested
   in-stock alternatives for anything unavailable.
4. Records an **auditable outcome** for every email and exports an operations report.

It runs as a **batch pipeline** (CSV in, Excel/CSV report out) and as an **HTTP API** that shares the same core.

---

## Quick start

**1. See it work in 60 seconds, without an API key** (bundled replay fixtures, in-memory store):

```bash
pip install -r requirements-dev.txt
make demo          # processes data/inbound_customer_emails.csv -> exports/
make eval          # scores the run against evals/gold_labels.json
```

**2. Run the API and PostgreSQL + pgvector in Docker:**

```bash
cp .env.example .env              # set POSTGRES_PASSWORD (and OPENAI_API_KEY for live mode)
docker compose up -d --build      # API on http://localhost:8000, docs at /docs

curl -s localhost:8000/v1/emails -H 'content-type: application/json' -d '{
  "email_id": "E007",
  "body": "Hi, this is Liz. Please send me 5 CLF2109 Cable Knit Beanies and 2 pairs of FZZ1098 Fuzzy Slippers."
}'
```

**3. Use the live model** (`gpt-6-luna` by default): set `OPENAI_API_KEY` in `.env`. `LLM_MODE=auto` switches
to live mode automatically. To capture real model answers as the replay fixtures and score them:

```bash
make record        # live run over the sample emails, saves fixtures/llm_replays.json
make eval          # now measures the real model against the gold labels
```

Other entry points: `make run` (batch with your configured mode), `make serve` (API without Docker),
`make docker-run` (batch in Docker), `make docker-test` (unit + integration tests in Docker), `make help`.

---

## Architecture

```mermaid
flowchart LR
    IN["Email<br/>(CSV batch or POST /v1/emails)"] --> G["Guardrail<br/>injection screen, PII masking,<br/>quarantine tags"]
    G -- flagged --> ESC["Escalate to a human<br/>(no model call, no stock)"]
    G --> C["Intent classifier<br/>Structured Outputs"]
    C -- order request --> R1["Hybrid retrieval<br/>SKU/name mentions + dense/BM25 RRF"]
    C -- product inquiry --> R2["Hybrid retrieval"]
    R1 --> X["Order extractor<br/>Structured Outputs over candidates,<br/>IDs re-validated against catalog"]
    X --> RES["Atomic reservation<br/>row locks, idempotent per email"]
    RES --> W1["Order reply<br/>from verified facts"]
    R2 --> W2["Inquiry reply<br/>grounded in retrieved products"]
    W1 --> V["Reply validation + framing<br/>grounding, placeholders, PII restore,<br/>greeting & signature"]
    W2 --> V
    V --> OUT["Outcome + audit trail<br/>API response / Excel & CSV report"]
    RES <--> DB[("PostgreSQL 16 + pgvector<br/>or in-memory store")]
    R1 <--> DB
    R2 <--> DB
```

Every model call has a deterministic fallback (rules or templates). When the model times out, is unavailable,
returns invalid output or a reply fails validation, the email is still handled and the outcome records why
it was degraded.

### How a batch is processed

| Phase | Concurrency | Work |
|---|---|---|
| 1. Analyze | parallel (`BATCH_WORKERS`) | guardrail, PII masking, classification, retrieval, order extraction |
| 2. Commit | **sequential, in arrival order** | idempotent claim of the email ID and atomic stock reservation |
| 3. Respond | parallel | grounded reply generation, validation, audit record |

Phase 2 makes stock allocation first-come-first-served and reproducible, while the slow model calls run
concurrently. A single API request runs the same three phases inline.

---

## Design decisions

- **The model never decides facts.** It classifies, maps the email onto a short list of candidate products and
  writes prose. Product IDs it returns are re-validated against the catalog (with typo correction).
  Unknown items become *unresolved* and the customer is asked to clarify; nothing is guessed. Stock
  decisions are made in code, inside a database transaction.
- **Replies are validated before sending.** The model writes only the body. Greeting, sign-off and
  signature come from configuration. A reply is rejected, and replaced by a factual template, if it cites a
  product ID outside its grounded context, omits an ordered product, contains a placeholder such as
  `[Your Name]`, or leaves an unknown redaction token.
- **Retrieval separates "mentioned" from "related".** Products the customer explicitly referenced (SKU,
  typo'd SKU or name) are always in context; up to *k* related products are added by
  Reciprocal Rank Fusion of dense vector search and BM25. Only these products go into prompts, so prompt size does
  not grow with catalog size.
- **Both storage backends rank identically.** PostgreSQL uses its indexes (HNSW for vectors, GIN full-text,
  `pg_trgm` for SKUs) only to *generate candidates*; scoring is shared Python code. An integration test checks
  ranking parity, which also keeps replay fixtures valid on either backend.
- **All-or-nothing per order line.** If stock is insufficient for the requested quantity, the line is
  `out of stock` and stock is untouched. Concurrent orders lock product rows in primary-key order
  (`SELECT ... FOR UPDATE`) so they cannot oversell or deadlock.
- **Idempotent processing.** The email ID is claimed in the same transaction as the reservation. A
  re-delivered email returns its stored outcome. An email interrupted after reserving resumes without
  reserving again.
- **Security and privacy.** Emails, phone numbers, card/account numbers, SSNs and street addresses are
  replaced by tokens before any model call and restored only in the outbound reply. Customer text sits inside
  `<customer_untrusted_input>` tags. Emails matching injection signatures are escalated without a model
  call. The database stores outcomes, not raw email bodies, and logs never contain email content.
- **Deterministic replay.** Every request is fingerprinted (model + prompt + input). Replays are strict: if a
  prompt changes, the replay misses rather than returning a stale answer, and a test fails until fixtures
  are rebuilt.

### LLM modes (`LLM_MODE`)

| Mode | Behaviour |
|---|---|
| `auto` (default) | `live` if `OPENAI_API_KEY` is set, else `replay` if fixtures exist, else `offline` |
| `live` | OpenAI chat model (`LLM_MODEL`, default `gpt-6-luna`) with Structured Outputs + `EMBEDDING_MODEL` |
| `record` | `live`, and every answer and embedding is saved to `REPLAY_FIXTURES_PATH` |
| `replay` | answers come from the fixture file; no network, fully deterministic |
| `offline` | no model at all: rule-based classification/extraction and templated replies |

---

## Evaluation

`evals/gold_labels.json` holds hand labels for all 23 sample emails: intent, exact order lines (SKU,
quantity, status), clarification status, customer name, reply language and the products a reply must
reference. `python -m src.cli eval` runs the whole pipeline on a fresh store and scores it. CI fails if
intent accuracy or order-line recall drops below 90%.

| Metric | Replay fixtures | Offline rules |
|---|---|---|
| Intent accuracy | 100% | 100% |
| Order-line precision / recall | 100% / 100% | 100% / 100% |
| Orders fully correct | 100% | 100% |
| Customer name accuracy | 100% | 100% |
| Inquiry context recall | 100% | 100% |
| Replies without placeholder leaks | 100% | 100% |

Full report: [evals/RESULTS.md](evals/RESULTS.md).

**How to read these numbers:**
- The bundled fixtures are **hand-written reference answers** (`fixtures/reference_answers.json`), bound to the
  exact prompts by `scripts/build_reference_fixtures.py`. The fixture file is labelled `source: hand-authored`.
  Scoring them verifies retrieval, SKU resolution, reservation logic, reply validation and reporting
  end to end, but **not the model's own accuracy**. Run `make record && make eval` with an API key to measure
  `gpt-6-luna` on the same gold labels. The file is then relabelled `source: recorded`.
- The offline rules were developed against these same 23 emails, so their score is not a held-out
  estimate. The unit tests include additional held-out phrasings.

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness |
| `GET` | `/readyz` | Readiness (storage reachable) |
| `POST` | `/v1/emails` | Process one email `{email_id?, subject?, body}`; idempotent on `email_id` |
| `POST` | `/v1/emails/batch` | Process up to `API_MAX_BATCH` emails with the three-phase batch engine |
| `GET` | `/v1/emails/{email_id}` | Stored outcome |
| `GET` | `/v1/products/{product_id}` | Product with live stock |
| `GET` | `/v1/products?query=...&limit=5` | Hybrid search (mentioned + related) |

When `API_KEY` is set, `/v1` routes require the `X-API-Key` header. Every response carries `X-Request-ID`,
which also appears in the logs. Interactive docs are at `/docs`.

<details>
<summary>Example outcome</summary>

```json
{
  "email_id": "E007",
  "category": "order request",
  "status": "processed",
  "customer_name": "Liz",
  "lines": [
    {"product_id": "CLF2109", "product_name": "Cable Knit Beanie", "requested_quantity": 5,
     "fulfilled_quantity": 0, "status": "out of stock", "remaining_stock": 2},
    {"product_id": "FZZ1098", "product_name": "Fuzzy Slippers", "requested_quantity": 2,
     "fulfilled_quantity": 2, "status": "created", "remaining_stock": 0}
  ],
  "context_product_ids": ["CLF2109", "FZZ1098", "CHN0987", "SFT1098"],
  "degraded": false,
  "usage": {"llm_calls": 3, "replayed_calls": 3, "prompt_tokens": 0, "completion_tokens": 0},
  "response": "Dear Liz,\n\nThank you for your order! ..."
}
```
</details>

---

## Batch outputs

`make demo` / `make run` write `exports/commerceops_operations_report.xlsx` plus one CSV per sheet in
`exports/csv/`:

| Sheet | Columns |
|---|---|
| `email-classification` | email ID, category |
| `order-status` | email ID, product ID, quantity, status (`created` / `out of stock`) |
| `order-response` | email ID, response |
| `inquiry-response` | email ID, response |
| `processing-audit` | status, confidence, language, customer name, context products, guardrail flags, degraded reasons, LLM calls and tokens, latency |

The committed `exports/` were produced by `make demo`.

---

## Configuration

All settings are environment variables (or `.env`); see [.env.example](.env.example). Blank values fall back to
defaults. The most important ones:

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODE` | `auto` | see the table above |
| `OPENAI_API_KEY` | — | required for `live` / `record` |
| `LLM_MODEL` | `gpt-6-luna` | must support Structured Outputs |
| `LLM_TEMPERATURE` | unset | only sent when set |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | `text-embedding-3-small` / `1536` | the DB column is `vector(1536)` |
| `STORAGE_BACKEND` | `memory` | `postgres` in Docker Compose |
| `DATABASE_URL` or `POSTGRES_*` | — | connection settings |
| `BATCH_WORKERS` | `4` | parallelism for the analyze/respond phases |
| `API_KEY` | unset | enables API-key auth |
| `BRAND_NAME`, `AGENT_NAME`, `SUPPORT_EMAIL`, `PHONE_NUMBER` | Aura Fashion Atelier… | used in prompts, templates and signatures |
| `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M` | unset | adds an estimated cost to run summaries |
| `LOG_FORMAT` | `text` | `json` for structured logs |

---

## Development

```bash
make dev                  # install runtime + dev dependencies (Python 3.12+)
make check                # ruff, mypy, pytest with coverage (what CI runs)
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/commerceops_test make test-integration
make fixtures             # rebuild replay fixtures after changing prompts or retrieval
```

- **Tests:** 102 tests, 94% line coverage. The model path is exercised with a scripted client: routing,
  grounding, outages, injection escalation, hallucinated IDs and names, idempotency and per-email failure
  isolation. Also covered: guardrails, rules, retrieval, storage concurrency, templates, LLM clients and the
  API.
- **Integration tests** (`-m integration`) run against real PostgreSQL + pgvector. They cover migrations, the
  HNSW query plan, retrieval parity with the in-memory backend, 20 concurrent orders that must not oversell,
  and persisted idempotent outcomes. The target database is wiped.
- **CI** (`.github/workflows/ci.yml`): lint, types, unit tests, both evaluations with thresholds,
  integration tests against a pgvector service container, and a Docker build plus API smoke test.

### Project layout

```
src/
  api/            FastAPI app and schemas
  llm/            LLM interface, OpenAI client, replay/record
  pipeline/       guardrails, classifier, extractor, responder, templates, rules, orchestrator
  retrieval/      text normalization, SKU/name matching, BM25, dense ranking, hybrid retriever
  storage/        memory and PostgreSQL stores, migrations, connection pool
  catalog.py      catalog sync (re-embeds only changed products)
  engine.py       wiring shared by CLI, API and evaluation
  evaluation.py   gold-label scoring
  cli.py          run | eval | migrate | seed | serve
data/             sample catalog (99 products) and inbound emails (23)
evals/            gold labels and results
fixtures/         reference answers and replay fixtures
tests/            unit and integration tests
```

---

## Limitations

- Fallback templates (used offline or when a reply fails validation) are written in English. Only the
  greeting and sign-off are localized (en/es/fr). Model-written replies follow the customer's language.
- Offline and hand-authored fixtures use a local feature-hashing embedder, which is lexical rather than
  semantic. Live mode uses the configured embedding model.
- Partial fulfilment is intentionally not supported: each line is reserved in full or not at all.
- There is no mailbox connector (IMAP, Gmail, helpdesk webhooks). The HTTP API is the integration point.
- The prompt-injection screen is signature-based. It is a first line of defence; the main safeguard is that
  the model cannot act on its own: stock, prices and product IDs are all decided and verified in code.

---

## Author & license

Created by [boucoding](https://github.com/boucoding) ·
[LinkedIn](https://www.linkedin.com/in/aadiab)

Licensed under [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(see [LICENSE](LICENSE)): free for personal, educational and research use; commercial use requires the
author's written consent.
