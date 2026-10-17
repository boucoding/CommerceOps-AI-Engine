# Evaluation results

Generated with `make eval` and `make eval-offline` against [gold_labels.json](gold_labels.json) (23 sample emails).

> **Reading these results.** The bundled replay fixtures are hand-written reference answers
> (`source: hand-authored`), so the replay score validates the pipeline around the model
> (retrieval, SKU resolution, reservations, reply validation, reporting), not the model itself.
> Run `make record && make eval` with an `OPENAI_API_KEY` to measure the configured model.
> The offline rules were developed against these same emails, so their score is not a held-out estimate.

## Evaluation: replay (hand-authored fixtures)

| Metric | Score |
|---|---|
| intent accuracy | 100.0% |
| order line precision | 100.0% |
| order line recall | 100.0% |
| orders fully correct | 100.0% |
| email status accuracy | 100.0% |
| customer name accuracy | 100.0% |
| reply language accuracy | 100.0% |
| inquiry context recall | 100.0% |
| replies without placeholder leaks | 100.0% |
| replies with brand signature | 100.0% |

| Count | Value |
|---|---|
| emails | 23 |
| degraded (fell back to rules/templates) | 0 |
| LLM calls | 59 |
| prompt tokens | 0 |
| completion tokens | 0 |

### Mismatches
- none


## Evaluation: offline

| Metric | Score |
|---|---|
| intent accuracy | 100.0% |
| order line precision | 100.0% |
| order line recall | 100.0% |
| orders fully correct | 100.0% |
| email status accuracy | 100.0% |
| customer name accuracy | 100.0% |
| reply language accuracy | 100.0% |
| inquiry context recall | 100.0% |
| replies without placeholder leaks | 100.0% |
| replies with brand signature | 100.0% |

| Count | Value |
|---|---|
| emails | 23 |
| degraded (fell back to rules/templates) | 0 |
| LLM calls | 0 |
| prompt tokens | 0 |
| completion tokens | 0 |

### Mismatches
- none

