# Dimension 1 — Retrieval Evaluation Protocol

Implementation: `evaluation/retrieval_eval.py` · CLI: `scripts/run_retrieval_evaluation.py`
· Tests: `tests/test_retrieval_eval.py`

## What is evaluated

Three configurations, all built from the **production** `HybridRetriever`
(`retrieval/hybrid_retriever.py`) over the production `FAISSStore`,
`BM25Retriever`, `CohereReranker` and the production chunk identity function
`retrieval.identity.chunk_identity_key`. No ranking, fusion or reranking logic
is re-implemented here.

| Config key       | Label                    | How it is built |
|------------------|--------------------------|-----------------|
| `dense`          | Dense Only               | `HybridRetriever(dense, <BM25 stub returning nothing>, embedder, use_reranker=False)`. Production RRF then fuses a single list, which preserves the dense order. |
| `dense_bm25`     | Dense + BM25             | `HybridRetriever(dense, bm25, embedder, use_reranker=False)` (production dense + BM25 + RRF, `k=60`). |
| `hybrid_cohere`  | Hybrid + Cohere Reranker | `HybridRetriever(dense, bm25, embedder, use_reranker=True)` — the exact production pipeline (`rerank-english-v3.0`). |

Every configuration calls `retrieve(question, top_k_retrieve=50, top_k_rerank=10)`.
50 is the production candidate pool; 10 is the number of ranked chunks scored.
`top_k_rerank=10` only changes how many results are *returned* (Cohere's
`top_n`), so it does not change which chunks occupy ranks 1–5.

Retrieved chunks are mapped to identity keys with `chunk_identity_key` — the
same function used to compute `gold_chunk_ids` in `benchmark.json`
(`text-hash:<sha256>` for the committed index, which predates `chunk_id`).
Repeated keys in a ranking are collapsed, keeping the first occurrence.

## Ground truth and scoring set

Ground truth is `gold_chunk_ids`. Relevance is binary: a retrieved chunk is
relevant iff its identity key is in the question's `gold_chunk_ids`.

**Scorable questions** are those with a non-empty `gold_chunk_ids`
(`evidence_status == "sufficient"`). The loader enforces this invariant and
refuses inconsistent data (sufficient with no gold, or insufficient/ambiguous
with gold).

**Excluded from all metrics** (retrieval metrics are undefined without a gold
chunk): `unanswerable` (insufficient), `ambiguous` (only document ids, no chunk
truth), and `adversarial` questions with `evidence_status == "insufficient"`.
They are never sent to the retriever. Their counts are reported per category
in the JSON report (`counts.excluded_by_category`,
`counts.excluded_by_evidence_status`) and in the table header.
Whether the *system* correctly abstains on these questions is a
generation-side property and is **out of scope** for retrieval evaluation.
Adversarial questions that do have gold chunks (the answerable/injection ones)
are scored like any other; the injected fixtures matter only for generation.

## Metrics

Per question, then macro-averaged over scorable questions:

* **Recall@k** (k = 5, 10) = |gold ∩ top-k| / |gold|. For multi-document
  questions (two gold chunks) retrieving one scores 0.5, both scores 1.0.
  The report also includes the strict `all_gold@k` (1.0 only if *every* gold
  chunk is in the top-k) as a supplementary figure.
* **MRR** = 1 / rank of the *first* gold chunk in the ranking, 0 if none is in
  the returned top 10 — i.e. it is **MRR@10**, identical depth for all three
  configurations.
* **nDCG@5** with binary gains and a `log2(rank+1)` discount. The ideal
  ranking places `min(|gold|, 5)` relevant chunks first, so multi-document
  questions only reach 1.0 when all gold chunks are on top.

The JSON report additionally contains a 95% percentile-bootstrap CI for each
metric (2000 resamples, seed 42), a per-category breakdown, and per-question
results (gold ids, retrieved ids, per-question scores).

## No tuning; held-out discipline

* The module exposes **no hyperparameters**: pool size (50), depth (10) and
  RRF `k` (60, read from the production default) are fixed. Nothing is fitted
  on any split; the reported configurations are the production defaults.
* `calibration.json` is the only split intended for calibration/development
  runs (default `--split calibration`).
* `test.json` is refused unless `--confirm-final-test` is passed, and this
  check runs before any index or model is loaded. Run it once, at the end.
* `--split` accepts only `calibration` or `test` — the combined
  `benchmark.json` is deliberately not selectable.

## Safeguards against misleading numbers

* If the Cohere reranker silently falls back to position order (missing/invalid
  key, API error), the run **aborts** with `RerankerFallbackError` instead of
  reporting Dense+BM25 results as "Hybrid + Cohere Reranker".
* Before running, all gold chunk ids are checked against the loaded corpus
  (production identity function); a rebuilt index that no longer matches the
  benchmark aborts the run.
* FAISS index size, metadata length and BM25 corpus length must agree.
* A Git LFS pointer in place of `faiss_index.bin` / `bm25_index.pkl` is
  detected with an actionable message (`git lfs pull`).
* The report records the split file path and SHA-256 and the protocol used.

## Running

Requirements: the real `data/processed/faiss_index.bin`,
`faiss_metadata.json`, `bm25_index.pkl` (not LFS pointers), the benchmark
files in `data/benchmark/` (or `--benchmark-dir`), the `BAAI/bge-large-en-v1.5`
model available to `sentence-transformers`, and `COHERE_API_KEY` (env or `.env`)
for the Cohere configuration.

```bash
# Development / smoke run on the calibration split (all three configs)
python scripts/run_retrieval_evaluation.py --split calibration

# Without Cohere credentials (first two configs only)
python scripts/run_retrieval_evaluation.py --split calibration --configs dense dense_bm25

# FINAL, one-shot evaluation on the held-out split
python scripts/run_retrieval_evaluation.py --split test --confirm-final-test
```

Output: a Markdown table on stdout and the full report at
`data/processed/retrieval_eval_<split>.json`.

```
| Configuration | Recall@5 | Recall@10 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense Only | … | … | … | … |
| Dense + BM25 | … | … | … | … |
| Hybrid + Cohere Reranker | … | … | … | … |
```

Runtime note: the production `CohereReranker` self-throttles to one call per
6.5 s (free tier), so the Cohere configuration takes roughly
`6.5 s × (number of scorable questions)` (≈ 4 min for calibration,
≈ 17 min for test).

## Tests

`pytest tests/test_retrieval_eval.py` — metric unit tests (Recall@5/10, MRR,
nDCG@5, multi-gold, duplicates, empty gold), loader/exclusion tests, all three
configurations on a synthetic corpus using the real FAISS/BM25/RRF/Cohere-client
code paths, and CLI end-to-end tests. Cohere is mocked (the production
`CohereReranker` is used with a fake client) and sockets are blocked in every
test, so no real external call can occur. Two integrity tests run against the
real benchmark files when present (splits disjoint and covering; every gold id
exists in `faiss_metadata.json`).

## Known limitations

* Reference answers and gold chunks are LLM-generated and LLM-validated, not
  human-annotated (see benchmark metadata). Recall is measured against that
  gold, which can under-count other chunks that also answer the question.
* Identity keys are text hashes; 17 corpus chunks share text with another
  chunk (7,475 unique keys for 7,492 chunks), so a duplicate of a gold chunk
  counts as the same chunk.
* MRR is truncated at 10.
