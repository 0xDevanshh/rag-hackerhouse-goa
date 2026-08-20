# Retrieval evaluation — ai4bharat/MSMARCO-XI

Retrieval-only metrics scored against the dataset's own `is_selected` relevance
labels. A passage counts as retrieved once, regardless of how many languages it
was indexed in. No LLM is involved, so these numbers are deterministic.

Cases: **102** | MRR: **0.483** | low-confidence rate: **0.0%**

Coverage: 0.0% of cases returned fewer than 10 distinct passages, so their metrics at the largest k are bounded by retrieval depth rather than ranking quality.

## Overall

| k | hit rate@k | recall@k | precision@k |
| --- | --- | --- | --- |
| 1 | 30.4% | 29.9% | 30.4% |
| 3 | 54.9% | 53.9% | 18.6% |
| 5 | 68.6% | 67.2% | 14.1% |
| 10 | 92.2% | 90.2% | 9.6% |

## Label-leak control

The same eval re-run with `Retriever(is_selected_boost=0.1)` — the production
default. `is_selected` is the relevance label being scored, so boosting by it
hands the ranker the answer. The gap below is the size of that leak, and is the
reason the headline numbers above disable the boost.

| metric | honest | with is_selected boost |
| --- | --- | --- |
| MRR | 0.483 | 0.483 |
| hit rate@1 | 30.4% | 30.4% |
| hit rate@3 | 54.9% | 54.9% |
| hit rate@5 | 68.6% | 68.6% |
| hit rate@10 | 92.2% | 92.2% |

## By query language

| language | n | MRR | hit@1 | hit@3 | hit@5 | hit@10 |
| --- | --- | --- | --- | --- | --- | --- |
| en | 51 | 0.594 | 41.2% | 70.6% | 84.3% | 98.0% |
| hi | 51 | 0.373 | 19.6% | 39.2% | 52.9% | 86.3% |
