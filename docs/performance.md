# D5-Lite: RAG performance and token/cost measurement

D5-Lite measures **where time goes** and **how many tokens a query uses** in the
RAG pipeline, and turns that into an *estimated* cost. It is deliberately small.
It is not observability infrastructure: no Prometheus, Grafana, distributed
tracing, authentication or RBAC.

It changes no retrieval, routing, evidence, security or generation behavior.
The instrumentation only wraps existing calls in timers and reads usage numbers
the LLM provider already returns.

**This document contains no performance numbers on purpose.** Run the benchmark
on your own deployment to get them (see "How to run").

## What it measures

| Measurement | Where it comes from |
|---|---|
| Per-stage latency, ms | `time.perf_counter()` around each stage (monotonic, high resolution) |
| Total end-to-end latency, ms | same clock, from the start of request handling to the response being built |
| Input / output / total tokens | the `usage` block the provider returns with each LLM call. **Never estimated or counted locally** |
| Estimated cost | reported tokens x the configured price per million tokens |
| Routing path | the path that produced the final chunks |

### Stages

| Response key | Stage | Notes |
|---|---|---|
| `routing_ms` | `QueryRouter.route` | adaptive routing only |
| `dense_retrieval_ms` | embed the query + vector search | embedding time is included |
| `bm25_retrieval_ms` | BM25 search | |
| `fusion_ms` | reciprocal rank fusion | also covers merging sub-query pools |
| `reranking_ms` | Cohere rerank (or the fallback) | |
| `rerank_throttle_wait_ms` | the client-side sleep in `CohereReranker` that paces the free tier to 10 calls/minute | **included inside `reranking_ms`**, shown separately so pacing is not mistaken for Cohere latency |
| `reformulation_ms` | the single retry's query rewrite | only when a retry happened; with `ADAPTIVE_REFORMULATION_MODE=llm` it includes the LLM call |
| `generation_ms` | the generation step, including its retry back-off sleeps | only when generation ran |
| `total_latency_ms` | whole request | includes unmeasured work (guards, evidence evaluation, prompt building) |

A stage that runs more than once in a request (dense/BM25/fusion once per
sub-query; reranking again on the retry) is **summed**.

Stages are independent measurements, **not a partition of the total**: the
throttle wait is inside reranking, and unnamed work exists only in the total.
Do not expect the stages to add up to `total_latency_ms`.

### Routing path

`routing_path` is the path that produced the chunks the answer was based on:

| Value | Meaning |
|---|---|
| `direct` | adaptive routing, one hybrid retrieval |
| `decomposed` | adaptive routing, sub-queries merged and reranked once |
| `reformulated` | the first attempt had weak evidence and the single retry ran (the original route stays in `routing.path`) |
| `full_retrieval` | `ADAPTIVE_ROUTING_ENABLED=false`: the legacy path. This is the **baseline** for comparisons |
| *(absent)* | the request was refused before retrieval (input guard) |

## Response schema

Backward compatible and **opt-in**. By default `/query` responses are
byte-for-byte what they were before D5-Lite. Send `"include_performance": true`
in the request, or set `PERFORMANCE_METADATA_ENABLED=true` to always include it:

```json
{
  "answer": "...",
  "performance": {
    "total_latency_ms": 0.0,
    "stages": {
      "routing_ms": 0.0,
      "dense_retrieval_ms": 0.0,
      "bm25_retrieval_ms": 0.0,
      "fusion_ms": 0.0,
      "reranking_ms": 0.0,
      "generation_ms": 0.0
    },
    "tokens": {"input": 0, "output": 0, "total": 0},
    "estimated_cost": 0.0,
    "cost_currency": "USD",
    "routing_path": "direct"
  }
}
```

(The zeros above are placeholders showing the shape, not measurements.)

Only fields that were actually measured appear:

* a stage that did not run is **absent**, never `0` (a refused query has no `generation_ms`);
* `tokens` is absent when the provider reported no usage; a *reported* zero stays `0`;
* `estimated_cost` is absent when tokens are missing or no price is configured;
* a request blocked by the input guard has only `total_latency_ms`.

When LLM reformulation is enabled and ran, `tokens` also contains
`by_purpose` (`reformulation` / `generation`). If some call made no usage
report, `tokens.incomplete` is `true` and the totals are a lower bound.

The block contains **no question, answer, document text or session id**. The
same data is logged as one JSON line on the `enterprise_rag.performance` logger
at INFO (silent unless you enable INFO for that logger, e.g.
`logging.getLogger("enterprise_rag.performance").setLevel(logging.INFO)`).

## Token accounting

Tokens are summed over the LLM calls made for the request: generation, plus the
reformulation call when `ADAPTIVE_REFORMULATION_MODE=llm` and a retry happened.
Retried generation attempts that raised an error are not counted (the provider
returned no usage for them), so tokens, and cost, can **under-count** real
provider usage when generation retries occurred.

## Cost estimation

> **Cost is an estimate, not billing data.** It is reported tokens x the price
> you configured. Free tiers, discounts, cached-token pricing, taxes, failed or
> retried calls, and Cohere reranking cost are **not** modelled. Only the LLM is priced.

Pricing is configuration, never code. Resolution order:

1. `LLM_PRICE_INPUT_PER_1M` + `LLM_PRICE_OUTPUT_PER_1M` (both or neither) and optional `LLM_PRICE_CURRENCY`;
2. the pricing file `config/llm_pricing.json` (or `LLM_PRICING_FILE`), entry `"<provider>/<model>"`, for example `groq/openai/gpt-oss-120b`, using the provider and model the generator is actually configured with;
3. otherwise no price: tokens are still reported, cost is omitted.

Malformed prices fail at startup with a message naming the variable.

The shipped entry for `groq/openai/gpt-oss-120b` carries an `as_of` date and an
**UNVERIFIED** note: it was taken from third-party price listings, which did not
fully agree on the output price. Confirm it against Groq's own pricing page and
edit the file (or set the environment variables) before relying on any cost
figure.

## How to run

```bash
# 1. start the API as usual
python app.py            # or: uvicorn api.rag_api:app --port 8000

# 2. in another terminal: benchmark (reuses existing benchmark questions)
python scripts/benchmark_performance.py --label adaptive --limit 20 \
    --exclude-categories adversarial --concurrency 1

# 3. summarize by routing path
python scripts/analyze_performance.py data/performance/benchmark_adaptive.json
```

Baseline comparison (full retrieval vs adaptive routing), same questions both times:

```bash
# restart the API with ADAPTIVE_ROUTING_ENABLED=false, then:
python scripts/benchmark_performance.py --label baseline --limit 20 --exclude-categories adversarial
# restart with ADAPTIVE_ROUTING_ENABLED=true (default), then:
python scripts/benchmark_performance.py --label adaptive --limit 20 --exclude-categories adversarial
python scripts/analyze_performance.py data/performance/benchmark_adaptive.json \
    --baseline data/performance/benchmark_baseline.json
```

Options worth knowing: `--concurrency N`, `--warmup N` (uncounted requests),
`--categories`, `--exclude-categories`, `--split` (`calibration` by default; `test`
is held out and needs `--confirm-final-test`), `--output`.

**Every benchmark request calls Groq and Cohere**, so it spends tokens and
counts toward their limits (including Groq's daily token cap and Cohere's
free-tier rate limit). The default `--limit` is 20 for that reason.

## Benchmark methodology

* **Questions** come from the existing Dimension 1 splits through its loader, in file order. `--split/--limit/--categories` always select the same questions. No question is written by hand.
* **Target** is the real running `/query` endpoint, so the numbers include everything the API does.
* **One fresh `session_id` per request**, so conversation memory cannot lengthen later prompts. The sessions are cleared afterwards.
* **Client latency** is wall time observed by the benchmark process (includes network and server queueing); **stage latency and tokens** are server-reported. Both are reported separately.
* **Throughput** is successful requests divided by wall-clock seconds from the first send to the last completion.
* **Percentiles** use linear interpolation between order statistics (the numpy default). P95 and P99 are flagged as unreliable at small sample sizes (n < 20 and n < 100).
* **Failed requests** (timeouts, HTTP errors, malformed responses) are counted and excluded from latency statistics; only the exception class or status code is recorded, never a response body.
* The report stores ids, categories, timings, token counts and cost. It stores no question or answer text.

## Limitations

* **The Cohere pacing sleep dominates reranking at low concurrency.** `CohereReranker` sleeps to stay under 10 calls/minute, so with the free tier reranking latency (and everything downstream, including throughput) mostly reflects that sleep, not Cohere or retrieval speed. Look at `rerank_throttle_wait_ms`; the analysis also reports latency excluding it. The pacing state is per process and not locked, so concurrent requests can sleep at the same time and may exceed the intended rate; with the free tier, higher `--concurrency` can trigger Cohere rate-limit errors (the reranker then falls back to unreranked order, which is marked `reranked: false`).
* **Not a load test.** No ramp-up, no distributed workers, no think-time model. Small samples give unstable P95/P99.
* **Results are specific to** your hardware, index, network, provider load, time of day and the Groq/Cohere account tier. Repeat runs vary.
* **Adaptive vs baseline comparisons are descriptive.** No significance test is applied. Differences in refusals between runs mix the routing effect with the refusal effect (a refused query makes no LLM call); the analysis therefore also gives a paired comparison over questions answered in both runs. Nothing here shows that adaptive routing saves money or time unless your measured numbers show it.
* **Cost covers the LLM only**, from provider-reported tokens, and is an estimate (see above). It can under-count when generation retries occurred.
* **Instrumentation overhead** is a few `perf_counter` calls per stage; it has not been separately measured.
* Dense and BM25 retrieval run sequentially; if they are ever parallelized, stage times will overlap and exceed the total.
* Adversarial benchmark questions are normally refused by the input guard before any retrieval, so they exercise almost none of the pipeline; the examples above exclude them.
* `/upload` and the Streamlit UI are not instrumented.

## Code map

```
performance/
  recorder.py   request-scoped stage timers + LLM usage (contextvar; no-op when inactive)
  tokens.py     provider usage parsing and aggregation
  pricing.py    PricingConfig (env / file) and estimate_cost
  config.py     PerformanceConfig.from_env
  report.py     the `performance` block and the log line
  stats.py      percentiles and summaries
config/llm_pricing.json              per-model prices (data, not code)
scripts/benchmark_performance.py     latency/throughput benchmark
scripts/analyze_performance.py       per-path and baseline-vs-adaptive analysis
```

Instrumented call sites: `retrieval/hybrid_retriever.py`, `retrieval/reranker.py`,
`retrieval/adaptive.py`, `retrieval/reformulator.py` and `api/rag_api.py`.
Tests: `tests/test_perf_*.py`.
