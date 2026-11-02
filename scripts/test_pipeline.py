import sys, os, pickle, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ingestion.embedder import CachedEmbedder
from retrieval.vector_store import QdrantStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from generation.llm_client import RAGGenerator

# Load components
embedder = CachedEmbedder()
qdrant = QdrantStore(collection="documents")

with open("data/processed/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

hybrid = HybridRetriever(qdrant, bm25, embedder)
generator = RAGGenerator(provider="anthropic")  # Free — use Claude

# Test queries
test_queries = [
    "What is retrieval augmented generation?",
    "How does dense retrieval work?",
    "What are the limitations of RAG systems?",
    "Explain the difference between BM25 and dense embeddings",
    "What evaluation metrics are used for RAG?",
]

print("=== END-TO-END PIPELINE TEST ===\n")

for query in test_queries:
    print(f"Q: {query}")
    chunks = hybrid.retrieve(query, top_k_retrieve=50, top_k_rerank=5)
    result = generator.generate(query, chunks)
    print(f"A: {result['answer'][:300]}...")
    print(f"   Sources retrieved: {len(chunks)}")
    print(f"   Top relevance score: {chunks[0].get('relevance_score', 'N/A')}")
    print()

print("Pipeline test complete. If all 5 answers look reasonable, you are ready for RAGAS.")