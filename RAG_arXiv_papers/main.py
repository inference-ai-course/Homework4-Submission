from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
import util


app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


chunks = []
faiss_index = None

@app.on_event("startup")
async def startup_event():
    """Load pdfs and generate chunks and create vector DB on startup"""
    
    try:
        # Fetch 50 cs.CL papers
        input_dir = 'arxiv_pdfs/'
        if not os.path.exists(input_dir):
            util.fetch_arxiv_papers(category='cs.CL', max_results=50)

        # Get all PDF files and generate chunks
        pdf_files = Path(input_dir).glob('*.pdf')
        for pdf_file in pdf_files:
            text = util.extract_text_from_pdf(pdf_file)
            temp_chunks = util.chunk_text(text)
            chunks.extend(temp_chunks)

        #Embedding and FAISS Indexing
        embeddings = util.embedding(chunks)
        global faiss_index
        faiss_index = util.faiss_indexing(embeddings)

    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        raise


@app.get("/")
async def root():
    """Serve the main UI"""
    return FileResponse("static/index.html")


@app.get("/api/search/")
async def search(q: str):
    print(f'user input: {q}')
    indexes = util.query(faiss_index, q)
    results = []
    for idx in indexes:
        results.append(chunks[idx])
    return {"query": q, "results": results}
    


