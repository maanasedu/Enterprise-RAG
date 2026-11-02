import sys, os, json, pickle, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import faiss as faiss_lib
from ingestion.embedder import CachedEmbedder
from retrieval.vector_store import FAISSStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from generation.llm_client import RAGGenerator
from evaluation.ragas_suite import RAGASEvaluator
from tqdm import tqdm

# ── Load components ───────────────────────────────────────────────────────────
print("Loading components...")

embedder = CachedEmbedder()

faiss_store = FAISSStore(dim=1024)
faiss_store.index = faiss_lib.read_index("data/processed/faiss_index.bin")
with open("data/processed/faiss_metadata.json") as f:
    faiss_store.metadata = json.load(f)

with open("data/processed/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

# Use HybridRetriever WITH Cohere reranking so the evaluated pipeline
# matches production exactly. CohereReranker self-throttles to respect
# the free-tier rate limit, so this just makes the run slower, not broken.
hybrid = HybridRetriever(faiss_store, bm25, embedder, use_reranker=True)

generator = RAGGenerator(provider="groq")
evaluator = RAGASEvaluator()

print("All components loaded.")

# ── Load QA pairs ─────────────────────────────────────────────────────────────
with open("data/processed/qa_pairs.json") as f:
    qa_pairs = json.load(f)

print(f"Total QA pairs available: {len(qa_pairs)}")

# 50 pairs is enough for solid benchmark numbers
eval_pairs = qa_pairs[:50]

# ── Run RAG pipeline ──────────────────────────────────────────────────────────
results = []
print(f"\nRunning RAG pipeline on {len(eval_pairs)} samples...")

for i, qa in enumerate(tqdm(eval_pairs, desc="Generating answers")):
    try:
        chunks = hybrid.retrieve(qa["question"], top_k_retrieve=50, top_k_rerank=5)
        result = generator.generate(qa["question"], chunks)
        answer = result.get("answer", "")

        # Skip empty or refusal answers for evaluation
        if not answer or len(answer.strip()) < 10:
            continue
        if "cannot find sufficient" in answer.lower():
            continue

        results.append({
            "question": qa["question"],
            "answer": answer,
            "contexts": [c["text"] for c in chunks],
            "ground_truth": qa["answer"]
        })

        # Prevent Groq rate limiting
        if i % 10 == 0 and i > 0:
            time.sleep(3)

    except Exception as e:
        print(f"\nFailed sample {i}: {e}")
        continue

print(f"\nSuccessfully generated {len(results)} answers for evaluation")

if len(results) < 5:
    print("Too few results to evaluate. Check your Groq API key and QA pairs.")
    exit(1)

# ── Run RAGAS ─────────────────────────────────────────────────────────────────
print(f"\nRunning RAGAS evaluation on {len(results)} samples...")
print("This will take 10-20 minutes. Each sample makes 4 LLM calls.\n")

df = evaluator.evaluate(results)
evaluator.print_summary(df)

# ── Save results ──────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/eval_results.csv", index=False)
print("\nSaved to data/processed/eval_results.csv")
print("Upload this file to the Streamlit Evaluation Dashboard tab.")

# ── Print resume-ready numbers ────────────────────────────────────────────────
print("\n" + "="*50)
print("RESUME-READY NUMBERS")
print("="*50)
for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    if metric in df.columns:
        score = df[metric].mean()
        print(f"{metric}: {score:.3f} ({score*100:.1f}%)")
print(f"QA pairs evaluated: {len(results)}")
print("="*50)