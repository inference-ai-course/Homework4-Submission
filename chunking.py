# Task 3: Text Chunking

#def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50):
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks

all_chunks = []

for paper in papers:
    full_text = paper.get("pdf_text", "")
    paper_id = paper["id"]
    print(f"Chunking {paper['id']}...")

    chunks = chunk_text(full_text, max_tokens=512, overlap=50)

    #paper["chunks"] = chunks

    # Convert each chunk into the structure expected by EmbeddingIndexer
    
    for i, ch in enumerate(chunks):
        all_chunks.append({
            "chunk_id": f"{paper_id}_chunk_{i}",
            "source_id": paper_id,
            "chunk_index": i,
            "token_count": len(ch.split()),
            "text": ch
        })

# Save as JSON
with open("paper_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Saved {len(all_chunks)} total chunks to paper_chunks.json")