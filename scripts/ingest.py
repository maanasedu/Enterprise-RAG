import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import json

load_dotenv()

from ingestion.parser import DocumentParser
from ingestion.chunker import DocumentChunker
from ingestion.embedder import CachedEmbedder
from retrieval.vector_store import QdrantStore, FAISSStore
from retrieval.bm25_retriever import BM25Retriever
import pickle

# Config
CHUNK_STRATEGY = "recursive"  # Best balance of speed and quality
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
DATA_DIR = "data/raw/arxiv"

# Initialize
parser = DocumentParser()
chunker = DocumentChunker()
embedder = CachedEmbedder()

# Qdrant
qdrant = QdrantStore(collection="documents")
qdrant.create_collection(dim=1024)

# FAISS backup
faiss_store = FAISSStore(dim=1024)

# BM25 corpus
bm25_corpus = []

print("Starting ingestion...")
pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
print(f"Found {len(pdf_files)} documents")

all_chunks = []

for pdf_path in tqdm(pdf_files, desc="Parsing + Chunking"):
    try:
        doc = parser.parse(str(pdf_path))
        chunks = chunker.chunk(
            doc.text,
            doc.metadata,
            strategy=CHUNK_STRATEGY,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        all_chunks.extend(chunks)
    except Exception as e:
        print(f"Failed {pdf_path.name}: {e}")
        continue

print(f"Total chunks created: {len(all_chunks)}")

# Embed in batches of 32
BATCH_SIZE = 32
all_embeddings = []

for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding"):
    batch = all_chunks[i:i + BATCH_SIZE]
    texts = [c.text for c in batch]
    embeddings = embedder.embed(texts)
    all_embeddings.extend(embeddings)

# NEW - sends in batches of 100
print("Indexing in Qdrant...")
QDRANT_BATCH_SIZE = 100
for i in tqdm(range(0, len(all_chunks), QDRANT_BATCH_SIZE), desc="Uploading to Qdrant"):
    batch_chunks = all_chunks[i:i + QDRANT_BATCH_SIZE]
    batch_embeddings = all_embeddings[i:i + QDRANT_BATCH_SIZE]
    qdrant.add(batch_embeddings, batch_chunks)

print("Indexing in FAISS...")
faiss_store.add(all_embeddings, all_chunks)

print("Building BM25 index...")
bm25_corpus = [{"text": c.text, "metadata": c.metadata} for c in all_chunks]
bm25 = BM25Retriever()
bm25.index(bm25_corpus)

# Save BM25 and FAISS to disk
with open("data/processed/bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

import faiss as faiss_lib
faiss_lib.write_index(faiss_store.index, "data/processed/faiss_index.bin")

with open("data/processed/faiss_metadata.json", "w") as f:
    json.dump(faiss_store.metadata, f)

print(f"""
Ingestion Complete:
  Documents processed: {len(pdf_files)}
  Total chunks: {len(all_chunks)}
  Embeddings stored: {len(all_embeddings)}
  Qdrant: populated
  FAISS: saved to disk
  BM25: saved to disk
""")