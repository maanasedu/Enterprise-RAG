import sys, os, json, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from tqdm import tqdm
import random

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

with open("data/processed/bm25_index.pkl", "rb") as f:
    import pickle
    bm25 = pickle.load(f)

# Sample 100 chunks
sampled_chunks = random.sample(bm25.corpus, min(100, len(bm25.corpus)))

qa_pairs = []

for chunk in tqdm(sampled_chunks, desc="Generating QA pairs"):
    text = chunk["text"]
    if len(text) < 100:
        continue

    prompt = f"""Given this text, generate 5 factual question-answer pairs.
Return ONLY a JSON array, no other text, no markdown:
[{{"question": "...", "answer": "..."}}]

Text: {text[:800]}"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            reasoning_effort='low'
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        pairs = json.loads(content)
        for pair in pairs:
            pair["source_chunk"] = text
            pair["metadata"] = chunk.get("metadata", {})
        qa_pairs.extend(pairs)
    except Exception as e:
        print(f"Failed chunk: {e}")
        continue

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/qa_pairs.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

print(f"\nGenerated {len(qa_pairs)} QA pairs")
print("Saved to data/processed/qa_pairs.json")