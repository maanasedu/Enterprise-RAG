import arxiv, os, time
from pathlib import Path

Path("data/raw/arxiv").mkdir(parents=True, exist_ok=True)

client = arxiv.Client()
search = arxiv.Search(
    query="retrieval augmented generation large language models",
    max_results=50,
    sort_by=arxiv.SortCriterion.Relevance
)

for i, paper in enumerate(client.results(search)):
    filename = f"data/raw/arxiv/paper_{i:03d}.pdf"
    paper.download_pdf(filename=filename)
    print(f"Downloaded {i+1}/50: {paper.title[:60]}")
    time.sleep(1)  # Be polite to arxiv

print("Done. 50 papers downloaded.")