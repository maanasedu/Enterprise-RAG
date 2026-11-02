import os
import json
import time
import pickle
import threading
import numpy as np
import faiss as faiss_lib
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from ingestion.parser import DocumentParser
from ingestion.chunker import DocumentChunker
from ingestion.embedder import CachedEmbedder
from retrieval.vector_store import FAISSStore, QdrantStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.adaptive import (
    AdaptiveConfig,
    build_adaptive_retriever,
    REFUSAL_MESSAGE,
    REFUSAL_REASON,
)
from generation.llm_client import RAGGenerator
from generation.context_budget import build_context
from security.guardrails import (
    Guardrails,
    SecurityConfig,
    SECURITY_REFUSAL_REASON,
)
from security.pipeline import build_chat_messages, build_system_prompt
from performance.config import PerformanceConfig
from performance.recorder import (
    GENERATION,
    PURPOSE_GENERATION,
    RequestRecorder,
    activate,
    deactivate,
    record_llm_usage,
    stage,
)
from performance.report import (
    ROUTING_PATH_FULL_RETRIEVAL,
    build_performance,
    log_performance,
    routing_path_from_adaptive,
)


app = FastAPI(
    title="Enterprise RAG API",
    version="2.0.0",
    description="Production RAG with conversation memory and document upload"
)


# ── Global state ──────────────────────────────────────────────────────────────
embedder = None
faiss_store = None
bm25 = None
hybrid = None
generator = None
# Adaptive-routing settings (Dimension 2), read from the environment once at
# startup. ADAPTIVE_ROUTING_ENABLED=false restores the exact legacy path.
adaptive_config = None
# Prompt-injection guardrails (Dimension 3), built once at startup.
# SECURITY_GUARDRAILS_ENABLED=false restores the exact pre-guardrail behavior.
security_guardrails = None
# Latency / token / cost metadata (D5-Lite): response opt-in + pricing config.
# Measurement always runs; the `performance` block is only added to a /query
# response on request (include_performance) or when
# PERFORMANCE_METADATA_ENABLED=true, so the default response shape is unchanged.
performance_config = None

conversation_store: dict = {}

# Guards the "stage everything, then commit" section of /upload so two
# uploads processed back-to-back can't interleave their commits. It does
# not make the whole request atomic (see _persist_index_files docstring
# for the remaining, documented limitation).
_upload_lock = threading.Lock()


def load_components():
    global embedder, faiss_store, bm25, hybrid, generator, adaptive_config
    global security_guardrails, performance_config

    print("Loading components...")

    embedder = CachedEmbedder()

    faiss_store = FAISSStore(dim=1024)

    try:
        faiss_store.index = faiss_lib.read_index(
            "data/processed/faiss_index.bin"
        )
        with open("data/processed/faiss_metadata.json") as f:
            faiss_store.metadata = json.load(f)

        print(f"FAISS: {faiss_store.index.ntotal} vectors loaded")

    except FileNotFoundError:
        print(
            "WARNING: FAISS index not found. "
            "Upload documents to get started."
        )

    try:
        with open("data/processed/bm25_index.pkl", "rb") as f:
            bm25 = pickle.load(f)

        print(f"BM25: {len(bm25.corpus)} documents loaded")

    except FileNotFoundError:
        bm25 = BM25Retriever()
        print("WARNING: BM25 index not found.")

    hybrid = HybridRetriever(
        faiss_store,
        bm25,
        embedder,
        use_reranker=True
    )

    generator = RAGGenerator(provider="groq")

    adaptive_config = AdaptiveConfig.from_env()
    print(
        f"Adaptive routing: "
        f"{'enabled' if adaptive_config.enabled else 'disabled'}"
    )

    security_guardrails = Guardrails(SecurityConfig.from_env())
    print(
        f"Security guardrails: "
        f"{'enabled' if security_guardrails.enabled else 'disabled'}"
    )

    performance_config = PerformanceConfig.from_env(
        provider=getattr(generator, "provider", None),
        model=getattr(generator, "model", None)
    )
    print(
        f"Performance metadata: "
        f"{'always on' if performance_config.include_in_response else 'opt-in per request'}; "
        f"pricing: {performance_config.pricing.source}"
    )

    print("All components loaded.")


load_components()


def _persist_index_files(staged_index, staged_metadata, staged_bm25):
    """Write the new FAISS/BM25 indexes to disk.

    Each file is written to a temporary path and then moved into place
    with os.replace(), which is atomic on POSIX filesystems — a crash or
    error mid-write leaves the previous, still-valid file in place instead
    of a truncated/corrupt one.

    Known limitation: this makes each *individual* file's write atomic; it
    does not make the update of all three files atomic as a single unit.
    A crash between the three os.replace() calls could still leave files
    from different upload "generations" paired together on disk (e.g. a
    new faiss_index.bin with the old bm25_index.pkl). Fully solving that
    would need a versioned manifest (e.g. a single directory swap or a
    "current generation" pointer file) — a larger change than this pass
    is meant to make. What this DOES fix is the previous behavior, where
    the live in-memory faiss_store/bm25 objects were mutated *before* any
    of this was attempted, so a failure partway through left the running
    process serving a different (dirtier) state than whatever was on disk.
    With the staging approach below, a failure at any point before the
    final in-memory swap leaves the live, currently-serving index exactly
    as it was before the request.
    """
    os.makedirs("data/processed", exist_ok=True)

    tmp_faiss = "data/processed/faiss_index.bin.tmp"
    tmp_meta = "data/processed/faiss_metadata.json.tmp"
    tmp_bm25 = "data/processed/bm25_index.pkl.tmp"

    faiss_lib.write_index(staged_index, tmp_faiss)

    with open(tmp_meta, "w") as f:
        json.dump(staged_metadata, f)

    with open(tmp_bm25, "wb") as f:
        pickle.dump(staged_bm25, f)

    os.replace(tmp_faiss, "data/processed/faiss_index.bin")
    os.replace(tmp_meta, "data/processed/faiss_metadata.json")
    os.replace(tmp_bm25, "data/processed/bm25_index.pkl")


# ── Pydantic Models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: str = "default"
    # D5-Lite: ask for the `performance` block (timings, tokens, estimated
    # cost) in this response. Off by default; the response is then unchanged.
    include_performance: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    latency_ms: float
    session_id: str
    turn_number: int
    # Adaptive-routing fields (Dimension 2). Only present in the response
    # when adaptive routing is enabled (see response_model_exclude_unset on
    # /query), so the legacy response shape is unchanged when it is off.
    refused: Optional[bool] = None
    refusal_reason: Optional[str] = None
    routing: Optional[dict] = None
    evidence: Optional[dict] = None
    # Guardrail events (Dimension 3). Present only when a guardrail actually
    # fired on this request, so normal responses keep their existing shape.
    security: Optional[dict] = None
    # Latency / token / cost metadata (D5-Lite). Present only when requested
    # (see performance_config); contains no question, answer or document text.
    performance: Optional[dict] = None


class ClearSessionRequest(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int
    total_vectors: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse,
    response_model_exclude_unset=True
)
async def query(request: QueryRequest):

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    if faiss_store.index.ntotal == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed. Please upload documents first."
        )

    start = time.time()
    recorder = RequestRecorder()

    def perf_fields(routing_path=None, refused=None):
        # Never let measurement break a response.
        try:
            perf = build_performance(
                recorder,
                routing_path=routing_path,
                pricing=performance_config.pricing if performance_config else None
            )
            log_performance(perf, refused=refused)
            wanted = request.include_performance or (
                performance_config is not None
                and performance_config.include_in_response
            )
            return {"performance": perf} if wanted else {}
        except Exception as e:
            print(f"Performance metadata failed (ignored): {e}")
            return {}

    session_id = request.session_id or "default"

    if session_id not in conversation_store:
        conversation_store[session_id] = []

    history = conversation_store[session_id]
    turn_number = len(history) + 1

    guard = security_guardrails
    security_events = []

    # Layer 1 (Dimension 3): refuse before spending any retrieval/LLM budget.
    # A blocked question is never stored in history, so it cannot poison
    # later turns.
    if guard is not None and guard.enabled:
        input_verdict = guard.check_input(request.question)
        if input_verdict.blocked:
            from security.guardrails import SECURITY_INPUT_MESSAGE
            return QueryResponse(
                answer=SECURITY_INPUT_MESSAGE,
                sources=[],
                latency_ms=round((time.time() - start) * 1000, 2),
                session_id=session_id,
                turn_number=turn_number,
                refused=True,
                refusal_reason=SECURITY_REFUSAL_REASON,
                security={"events": [input_verdict.to_dict()]},
                **perf_fields(refused=True)
            )

    routing_path = None
    perf_token = activate(recorder)

    try:
        # Extra response fields, populated only on the adaptive path.
        adaptive_fields = {}

        if adaptive_config is not None and adaptive_config.enabled:
            # Route -> retrieve -> rerank -> evaluate evidence (with at most
            # one reformulated retry). Generation is decided below.
            adaptive_result = build_adaptive_retriever(
                hybrid, adaptive_config, generator
            ).retrieve(
                request.question,
                top_k_retrieve=50,
                top_k_rerank=request.top_k
            )

            adaptive_fields = {
                "routing": adaptive_result.routing_dict(),
                "evidence": adaptive_result.evidence_dict(),
            }
            routing_path = routing_path_from_adaptive(adaptive_result)

            if adaptive_result.refused:
                # Evidence too weak even after the allowed retry: refuse
                # instead of generating from it. Not stored in the
                # conversation history (same as the "no documents" path).
                return QueryResponse(
                    answer=REFUSAL_MESSAGE,
                    sources=[],
                    latency_ms=round((time.time() - start) * 1000, 2),
                    session_id=session_id,
                    turn_number=turn_number,
                    refused=True,
                    refusal_reason=REFUSAL_REASON,
                    **perf_fields(routing_path, True),
                    **adaptive_fields
                )

            adaptive_fields["refused"] = False
            chunks = adaptive_result.chunks
        else:
            routing_path = ROUTING_PATH_FULL_RETRIEVAL
            chunks = hybrid.retrieve(
                request.question,
                top_k_retrieve=50,
                top_k_rerank=request.top_k
            )

        if not chunks:
            return QueryResponse(
                answer="No relevant documents found for your question.",
                sources=[],
                latency_ms=round((time.time() - start) * 1000, 2),
                session_id=session_id,
                turn_number=turn_number,
                **perf_fields(routing_path)
            )

        # Layer 2 (Dimension 3): retrieved text is untrusted. Strip
        # instruction-like sentences, drop chunks that are mostly payload.
        if guard is not None and guard.enabled:
            chunks, context_report = guard.sanitize_chunks(chunks)
            if context_report.flagged:
                security_events.append(context_report.to_dict())
            if not chunks:
                from security.guardrails import SECURITY_CONTEXT_MESSAGE
                return QueryResponse(
                    answer=SECURITY_CONTEXT_MESSAGE,
                    sources=[],
                    latency_ms=round((time.time() - start) * 1000, 2),
                    session_id=session_id,
                    turn_number=turn_number,
                    refused=True,
                    refusal_reason=SECURITY_REFUSAL_REASON,
                    security={"events": security_events},
                    **perf_fields(routing_path, True),
                    **adaptive_fields
                )

        context = _format_context_with_history(
            chunks,
            history,
            request.question
        )

        with stage(GENERATION):
            answer = _generate_with_history(
                request.question,
                context,
                history
            )

        if not answer or len(answer.strip()) < 5:
            answer = (
                "I was unable to generate a response. "
                "Please try again."
            )

        # Layer 4 (Dimension 3): never let the system prompt, the canary or
        # a credential leave in an answer.
        if guard is not None and guard.enabled:
            output_verdict = guard.check_output(
                answer,
                system_prompt=build_system_prompt(guard)
            )
            if output_verdict.flagged:
                security_events.append(output_verdict.to_dict())
                answer = output_verdict.answer
                if output_verdict.action == "blocked":
                    return QueryResponse(
                        answer=answer,
                        sources=[],
                        latency_ms=round((time.time() - start) * 1000, 2),
                        session_id=session_id,
                        turn_number=turn_number,
                        refused=True,
                        refusal_reason=SECURITY_REFUSAL_REASON,
                        security={"events": security_events},
                        **perf_fields(routing_path, True),
                        **adaptive_fields
                    )

        if security_events:
            adaptive_fields["security"] = {"events": security_events}

        conversation_store[session_id].append({
            "question": request.question,
            "answer": answer,
            "timestamp": time.time()
        })

        if len(conversation_store[session_id]) > 10:
            conversation_store[session_id] = (
                conversation_store[session_id][-10:]
            )

        return QueryResponse(
            answer=answer,
            sources=chunks,
            latency_ms=round((time.time() - start) * 1000, 2),
            session_id=session_id,
            turn_number=turn_number,
            **perf_fields(routing_path, adaptive_fields.get("refused")),
            **adaptive_fields
        )

    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        deactivate(perf_token)


def _format_context_with_history(chunks, history, current_question):
    # Budget depends on whether we also need room for conversation history
    # in the prompt (see _generate_with_history) — this mirrors the two
    # budgets the old code applied via raw string slicing, but now chunks
    # are kept whole: a chunk that doesn't fit is dropped entirely rather
    # than cut off mid-sentence.
    max_chars = 3000 if history else 5000
    context, _chunks_included, _truncated = build_context(chunks, max_chars=max_chars)
    return context


def _generate_with_history(
    question: str,
    context: str,
    history: list
) -> str:

    import time as time_module

    # Shared with scripts/evaluate_security.py. With guardrails disabled this
    # produces exactly the messages the endpoint built before Dimension 3;
    # enabled, it adds the security rules and fences the context as untrusted.
    messages = build_chat_messages(
        question,
        context,
        history,
        guardrails=security_guardrails
    )

    for attempt in range(3):
        try:
            response = generator.client.chat.completions.create(
                model=generator.model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                timeout=30,
                reasoning_effort="low"
            )

            # D5-Lite: provider-reported usage, recorded as soon as the call
            # returns (a billed response that fails to parse below still counts).
            record_llm_usage(PURPOSE_GENERATION, response)

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(
                f"Generation attempt {attempt + 1} failed: {e}"
            )

            if attempt < 2:
                time_module.sleep(2 * (attempt + 1))

    return (
        "I was unable to generate a response at this time. "
        "Please try again."
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):

    filename = file.filename or ""

    if not filename.lower().endswith(
        (".pdf", ".docx", ".doc")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported"
        )

    contents = await file.read()

    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 20MB."
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / filename

    try:
        with open(file_path, "wb") as f:
            f.write(contents)

        parser = DocumentParser()
        parsed = parser.parse(str(file_path))

        if not parsed.text.strip():
            raise ValueError(
                "Could not extract text from document"
            )

        chunker = DocumentChunker()

        chunks = chunker.chunk(
            parsed.text,
            parsed.metadata,
            strategy="recursive",
            chunk_size=512,
            chunk_overlap=100
        )

        if not chunks:
            raise ValueError(
                "No chunks created from document"
            )

        texts = [c.text for c in chunks]

        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = embedder.embed(batch)
            all_embeddings.extend(batch_embeddings)

        # Validate before touching any index — a failure here means the
        # live index (in memory or on disk) is never mutated.
        if len(all_embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count ({len(all_embeddings)}) does not match "
                f"chunk count ({len(chunks)})"
            )

        vecs = np.array(
            all_embeddings,
            dtype="float32"
        )

        if vecs.ndim != 2 or vecs.shape[1] != faiss_store.index.d:
            got_dim = vecs.shape[1] if vecs.ndim == 2 else "unknown"
            raise ValueError(
                f"Embedding dimension ({got_dim}) does not match "
                f"index dimension ({faiss_store.index.d})"
            )

        faiss_lib.normalize_L2(vecs)

        new_chunk_records = [
            {
                "text": c.text,
                "metadata": c.metadata
            }
            for c in chunks
        ]

        # ── Stage everything off to the side first ──────────────────────
        # None of this touches the live faiss_store/bm25 objects that are
        # currently serving queries, so a failure anywhere above or below
        # this point (and before the commit block) leaves the running
        # index exactly as it was before this request — the previous
        # version mutated faiss_store.index and bm25.corpus in place
        # *before* attempting to persist to disk, so a failure partway
        # through could leave the live process serving chunks that were
        # never (and, on failure, would never be) saved.
        staged_index = faiss_lib.clone_index(faiss_store.index)
        staged_index.add(vecs)

        staged_metadata = faiss_store.metadata + new_chunk_records

        staged_bm25 = BM25Retriever()
        staged_bm25.index(bm25.corpus + new_chunk_records)

        # ── Commit ───────────────────────────────────────────────────────
        with _upload_lock:
            _persist_index_files(staged_index, staged_metadata, staged_bm25)

            # Only swap the live, currently-serving state in after disk
            # persistence has succeeded. `faiss_store` and `bm25` are
            # mutated in place (not rebound) so `hybrid`, which holds
            # references to these same objects, sees the update
            # immediately without needing to be reconstructed.
            faiss_store.index = staged_index
            faiss_store.metadata = staged_metadata
            bm25.corpus = staged_bm25.corpus
            bm25.bm25 = staged_bm25.bm25

        # Qdrant is a secondary, optional index (not the live retrieval
        # path — see HybridRetriever, which only talks to FAISS + BM25).
        # Its failure is logged but must never roll back, or block, the
        # FAISS/BM25 commit above.
        try:
            qdrant = QdrantStore(collection="documents")

            existing = [
                c.name
                for c in qdrant.client.get_collections().collections
            ]

            if "documents" not in existing:
                qdrant.create_collection(dim=1024)

            qdrant.add(
                all_embeddings,
                chunks
            )

        except Exception as e:
            print(
                f"Qdrant update failed (non-critical): {e}"
            )

        return UploadResponse(
            success=True,
            message=f"Successfully indexed {filename}",
            chunks_added=len(chunks),
            total_vectors=faiss_store.index.ntotal
        )

    except Exception as e:

        if file_path.exists():
            file_path.unlink()

        print(f"Upload error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/clear_session")
async def clear_session(
    request: ClearSessionRequest
):

    if request.session_id in conversation_store:
        conversation_store.pop(request.session_id)

    return {
        "success": True,
        "message": f"Session {request.session_id} cleared"
    }


@app.get("/session/{session_id}")
async def get_session(session_id: str):

    history = conversation_store.get(
        session_id,
        []
    )

    return {
        "session_id": session_id,
        "turns": len(history),
        "history": history
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "components": {
            "faiss_vectors": (
                faiss_store.index.ntotal
                if faiss_store
                else 0
            ),
            "bm25_documents": (
                len(bm25.corpus)
                if bm25
                else 0
            ),
            "generator": "groq/openai/gpt-oss-120b",
            "embedder": "BAAI/bge-large-en-v1.5",
            "active_sessions": len(conversation_store)
        }
    }


@app.get("/")
async def root():

    return {
        "message": "Enterprise RAG API v2.0",
        "docs": "/docs",
        "health": "/health"
    }
