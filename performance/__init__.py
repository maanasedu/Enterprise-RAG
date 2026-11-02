"""D5-Lite: RAG-specific latency, token and cost measurement.

Deliberately small and dependency-free. It does NOT do infrastructure
observability (no Prometheus, tracing, auth or RBAC); see docs/performance.md.

    recorder.py  per-request stage timers + LLM usage collection (contextvar)
    tokens.py    reading provider-reported token usage; aggregation
    pricing.py   configurable per-model pricing and cost estimation
    config.py    PerformanceConfig.from_env (response opt-in + pricing)
    report.py    the `performance` response block and the structured log line
    stats.py     percentile / summary helpers shared by the scripts

This package never imports retrieval/, generation/ or api/ code except the
leaf module retrieval.env_config, so the pipeline can import it without cycles.
"""
