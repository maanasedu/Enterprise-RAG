import sys, os, json, pickle, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from ingestion.parser import DocumentParser
from ingestion.chunker import DocumentChunker
from ingestion.embedder import CachedEmbedder
from retrieval.vector_store import FAISSStore
from retrieval.bm25_retriever import BM25Retriever
from generation.llm_client import RAGGenerator
from evaluation.ragas_suite import RAGASEvaluator
from tqdm import tqdm
import pandas as pd

strategies = [
    "sentence",
    "semantic",
    "sliding_window"
]

with open("data/processed/qa_pairs.json") as f:
    qa_pairs = json.load(f)

# Use only 15 QA pairs per strategy — fast and sufficient
eval_pairs = qa_pairs[:15]

# Use only 5 documents — keeps embedding time short
pdf_files = list(Path("data/raw/arxiv").glob("*.pdf"))[:5]

parser = DocumentParser()
embedder = CachedEmbedder()
generator = RAGGenerator(provider="groq")
evaluator = RAGASEvaluator()

all_results = {}

for strategy in strategies:
    print(f"\n{'='*40}")
    print(f"Testing strategy: {strategy.upper()}")
    print(f"{'='*40}")

    try:
        chunker = DocumentChunker(
            embedding_model=embedder if strategy == "semantic" else None
        )

        all_chunks = []
        for pdf in pdf_files:
            try:
                doc = parser.parse(str(pdf))
                chunks = chunker.chunk(
                    doc.text, doc.metadata,
                    strategy=strategy,
                    chunk_size=512,
                    chunk_overlap=50
                )
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  Failed parsing {pdf.name}: {e}")
                continue

        print(f"  Chunks created: {len(all_chunks)}")

        if not all_chunks:
            print(f"  Skipping {strategy} — no chunks created")
            continue

        # Embed
        print(f"  Embedding {len(all_chunks)} chunks...")
        embeddings = embedder.embed([c.text for c in all_chunks])

        # Index in FAISS
        faiss = FAISSStore(dim=1024)
        faiss.add(embeddings, all_chunks)

        # BM25
        bm25 = BM25Retriever()
        bm25.index([{"text": c.text, "metadata": c.metadata} for c in all_chunks])

        # Simple hybrid retrieve (no Cohere)
        def retrieve(query, top_k=5):
            qvec = embedder.embed([query])[0]
            dense = faiss.search(qvec, top_k=top_k * 2)
            sparse = bm25.search(query, top_k=top_k * 2)
            seen, merged = set(), []
            for r in dense + sparse:
                key = r["text"][:50]
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            return merged[:top_k]

        # Evaluate
        results = []
        print(f"  Evaluating {len(eval_pairs)} QA pairs...")
        for i, qa in enumerate(tqdm(eval_pairs, desc=f"  {strategy}")):
            try:
                chunks = retrieve(qa["question"], top_k=5)
                result = generator.generate(qa["question"], chunks)
                results.append({
                    "question": qa["question"],
                    "answer": result["answer"],
                    "contexts": [c["text"] for c in chunks],
                    "ground_truth": qa["answer"]
                })
                if i % 5 == 0 and i > 0:
                    time.sleep(1)
            except Exception as e:
                print(f"  Failed QA: {e}")
                continue

        if results:
            df = evaluator.evaluate(results)
            means = df.mean(numeric_only=True)
            all_results[strategy] = means
            print(f"  context_precision: {means.get('context_precision', 0):.3f}")
            print(f"  faithfulness:      {means.get('faithfulness', 0):.3f}")
            print(f"  answer_relevancy:  {means.get('answer_relevancy', 0):.3f}")
            print(f"  context_recall:    {means.get('context_recall', 0):.3f}")

    except Exception as e:
        print(f"  Strategy {strategy} failed: {e}")
        continue
# Final comparison table
print("\n" + "="*60)
print("CHUNKING STRATEGY COMPARISON RESULTS")
print("="*60)

if all_results:

    # Previous results (skip recomputing)
    all_results["fixed"] = {
        "context_precision": 0.313,
        "faithfulness": 0.320,
        "answer_relevancy": 0.367,
        "context_recall": 0.393
    }

    all_results["recursive"] = {
        "context_precision": 0.287,
        "faithfulness": 0.307,
        "answer_relevancy": 0.347,
        "context_recall": 0.360
    }

    comparison = pd.DataFrame(all_results).T
    print(comparison.to_string())
    comparison.to_csv("data/processed/chunking_comparison.csv")
    print("\nSaved to data/processed/chunking_comparison.csv")

    # Find winner
    if "context_precision" in comparison.columns:
        best = comparison["context_precision"].idxmax()
        worst = comparison["context_precision"].idxmin()
        best_score = comparison.loc[best, "context_precision"]
        worst_score = comparison.loc[worst, "context_precision"]
        improvement = ((best_score - worst_score) / worst_score * 100)
        print(f"\nBest strategy:  {best} ({best_score:.3f} context_precision)")
        print(f"Worst strategy: {worst} ({worst_score:.3f} context_precision)")
        print(f"Improvement:    {improvement:.1f}%")
        print(f"\nResume bullet: '{best} chunking outperformed {worst} by {improvement:.0f}% on context precision'")