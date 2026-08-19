# Retrieval evaluation — ai4bharat/MSMARCO-XI

Retrieval-only metrics scored against the dataset's own `is_selected` relevance
labels. A passage counts as retrieved once, regardless of how many languages it
was indexed in. No LLM is involved, so these numbers are deterministic.

Cases: **496** | MRR: **0.423** | low-confidence rate: **0.0%**

Coverage: 0.0% of cases returned fewer than 10 distinct passages, so their metrics at the largest k are bounded by retrieval depth rather than ranking quality.

## Overall

| k | hit rate@k | recall@k | precision@k |
| --- | --- | --- | --- |
| 1 | 22.8% | 22.1% | 22.8% |
| 3 | 55.8% | 54.1% | 19.0% |
| 5 | 68.1% | 66.5% | 14.2% |
| 10 | 80.4% | 79.8% | 8.6% |

## Label-leak control

The same eval re-run with `Retriever(is_selected_boost=0.1)` — the production
default. `is_selected` is the relevance label being scored, so boosting by it
hands the ranker the answer. The gap below is the size of that leak, and is the
reason the headline numbers above disable the boost.

| metric | honest | with is_selected boost |
| --- | --- | --- |
| MRR | 0.423 | 0.717 |
| hit rate@1 | 22.8% | 62.5% |
| hit rate@3 | 55.8% | 78.8% |
| hit rate@5 | 68.1% | 84.3% |
| hit rate@10 | 80.4% | 88.5% |

## By query language

| language | n | MRR | hit@1 | hit@3 | hit@5 | hit@10 |
| --- | --- | --- | --- | --- | --- | --- |
| en | 248 | 0.523 | 32.3% | 67.3% | 81.0% | 92.3% |
| hi | 248 | 0.322 | 13.3% | 44.4% | 55.2% | 68.5% |
