"""
Complete RAG System for arXiv Papers with LLM Generation + Simple Web UI
Retrieval-Augmented Generation using Llama 3.2 3B Instruct

Author: Student
Date: 2025-11-04
"""

import os
import json
import logging
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import numpy as np
import arxiv
import PyPDF2
import time
from io import BytesIO
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import uvicorn
from huggingface_hub import login

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

hf_token = open("../hf_token.txt").read().strip()  
login(token=hf_token)

# Detect device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
    logger.info(f"Using CUDA: {device_name}")
elif torch.backends.mps.is_available():
    device = "mps"
    logger.info("Using Apple Silicon MPS")
else:
    device = "cpu"
    logger.info("Using CPU")

CONFIG = {
    "arxiv_category": "cs.CL",
    "num_papers": 50,
    "chunk_max_tokens": 512,
    "chunk_overlap": 50, 
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
    "device": device,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "faiss_index_path": "./faiss_index.bin",
    "chunks_metadata_path": "./chunks_metadata.json",
    "papers_cache_path": "./papers_cache.json",
}

# Global variables
embedding_model = None
llm_pipeline = None
faiss_index = None
chunks_data = []


# ============================================================================
# HTML TEMPLATE (Embedded in Python)
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>arXiv RAG System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .input-group {
            margin: 20px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #4CAF50;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .btn-search {
            background-color: #2196F3;
        }
        .btn-search:hover {
            background-color: #0b7dda;
        }
        #loading {
            display: none;
            color: #666;
            margin: 20px 0;
        }
        #results {
            margin-top: 30px;
        }
        .answer-box {
            background: #f0f8ff;
            border-left: 4px solid #4CAF50;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .source {
            background: #fff;
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .source-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .source-meta {
            color: #666;
            font-size: 14px;
            margin: 5px 0;
        }
        .source-text {
            color: #555;
            margin-top: 10px;
            font-size: 14px;
        }
        .chunk-result {
            background: #fff;
            border-left: 3px solid #2196F3;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .stats {
            background: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 14px;
        }
        .error {
            background: #ffebee;
            border-left: 4px solid #f44336;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            color: #c62828;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 arXiv RAG System</h1>
        <p>Ask questions about recent CS/NLP research papers. The system will search relevant papers and generate an answer using AI.</p>
        
        <div class="input-group">
            <label for="query">Your Question:</label>
            <input type="text" id="query" placeholder="e.g., What is the transformer architecture?" />
        </div>
        
        <button onclick="askQuestion()">Ask with LLM (Slower, Better)</button>
        <button class="btn-search" onclick="searchOnly()">Just Search (Faster)</button>
        
        <div id="loading">⏳ Processing... This may take 10-30 seconds for LLM generation...</div>
        <div id="results"></div>
    </div>

    <script>
        async function askQuestion() {
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('Please enter a question!');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            try {
                const response = await fetch(`/ask?q=${encodeURIComponent(query)}&k=3`);
                const data = await response.json();
                
                if (response.ok) {
                    displayRAGResults(data);
                } else {
                    displayError(data.detail || 'An error occurred');
                }
            } catch (error) {
                displayError('Failed to connect to server: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        async function searchOnly() {
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('Please enter a search query!');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            try {
                const response = await fetch(`/search?q=${encodeURIComponent(query)}&k=5`);
                const data = await response.json();
                
                if (response.ok) {
                    displaySearchResults(data);
                } else {
                    displayError(data.detail || 'An error occurred');
                }
            } catch (error) {
                displayError('Failed to connect to server: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function displayRAGResults(data) {
            let html = `<div class="stats">✅ Generated answer using ${data.num_sources} source(s)</div>`;
            
            html += `<div class="answer-box">
                <h3>Answer:</h3>
                <p>${escapeHtml(data.answer)}</p>
            </div>`;
            
            html += `<h3>Sources:</h3>`;
            data.sources.forEach((source, idx) => {
                html += `<div class="source">
                    <div class="source-title">${idx + 1}. ${escapeHtml(source.paper_title)}</div>
                    <div class="source-meta">
                        📄 arXiv ID: ${escapeHtml(source.paper_id)} | 
                        👥 ${escapeHtml(source.authors.join(', '))} | 
                        📅 ${escapeHtml(source.published)}
                    </div>
                    <div class="source-text">${escapeHtml(source.text_preview)}</div>
                </div>`;
            });
            
            document.getElementById('results').innerHTML = html;
        }
        
        function displaySearchResults(data) {
            let html = `<div class="stats">🔍 Found ${data.num_results} relevant chunk(s)</div>`;
            
            html += `<h3>Search Results:</h3>`;
            data.results.forEach((result, idx) => {
                html += `<div class="chunk-result">
                    <div class="source-title">${idx + 1}. ${escapeHtml(result.paper_title)}</div>
                    <div class="source-meta">
                        Score: ${result.score.toFixed(4)} | 
                        📄 ${escapeHtml(result.paper_id)} | 
                        👥 ${escapeHtml(result.paper_authors.join(', '))}
                    </div>
                    <div class="source-text">${escapeHtml(result.text)}</div>
                </div>`;
            });
            
            document.getElementById('results').innerHTML = html;
        }
        
        function displayError(message) {
            document.getElementById('results').innerHTML = 
                `<div class="error">❌ Error: ${escapeHtml(message)}</div>`;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Allow Enter key to submit
        document.getElementById('query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def split_text_into_chunks(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    tokens = text.split()
    
    if len(tokens) == 0:
        return []
    
    chunks = []
    step = max_tokens - overlap
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = " ".join(chunk_tokens)
        
        if chunk_text.strip():
            chunks.append(chunk_text)
        
        if i + max_tokens >= len(tokens):
            break
    
    return chunks


# ============================================================================
# CONTENT EXTRACTION 
# ============================================================================

def extract_from_html_version(paper_id: str) -> Optional[str]:
    """
    Method 1: Try to get HTML version of full paper.
    This is the newest arXiv feature.
    """
    html_url = f"https://export.arxiv.org/html/{paper_id}"
    
    try:
        response = requests.get(html_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Get main content
            main_content = soup.find('article') or soup.find('main') or soup.find('body')
            
            if main_content:
                text = main_content.get_text()
            else:
                text = soup.get_text()
            
            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk) 

            
            if len(text) > 1000:  # Got substantial content
                logger.info(f"  ✓ HTML version: {len(text)} chars")
                return text
                
    except Exception as e:
        pass
    
    return None


def extract_from_pdf(paper_id: str) -> Optional[str]:
    """
    Method 2: Download and extract text from PDF.
    More reliable than HTML, works for almost all papers.
    """
    pdf_url = f"https://export.arxiv.org/pdf/{paper_id}.pdf"
    
    try:
        response = requests.get(pdf_url, timeout=60)
        if response.status_code == 200:
            # Read PDF from bytes
            pdf_file = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text_pages = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
            
            full_text = "\n\n".join(text_pages)
            
            if len(full_text) > 1000:  # Got substantial content
                logger.info(f"  ✓ PDF extraction: {len(full_text)} chars")
                return full_text
                
    except Exception as e:
        logger.warning(f"  ✗ PDF extraction failed: {e}")
    
    return None


def get_paper_metadata_from_abstract_page(paper_id: str) -> Optional[Dict]:
    """
    Get metadata (title, authors, abstract) from the abstract page.
    Your approach - this always works!
    """
    try:
        url = f"https://arxiv.org/abs/{paper_id}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title_elem = soup.find('h1', class_='title')
            abstract_elem = soup.find('blockquote', class_='abstract')
            authors_elem = soup.find('div', class_='authors')
            
            metadata = {
                'title': title_elem.text.replace('Title:', '').strip() if title_elem else '',
                'abstract': abstract_elem.text.replace('Abstract:', '').strip() if abstract_elem else '',
                'authors': [a.text for a in authors_elem.find_all('a')] if authors_elem else []
            }
            
            return metadata
    except:
        pass
    
    return None


def fetch_paper_with_fallback(paper_id: str, title: str, authors: List[str], 
                              summary: str, published: str) -> Dict:
    """
    Robust content fetching with multiple fallback methods.
    
    Strategy:
    1. Try HTML version (newest, cleanest)
    2. Try PDF extraction (most reliable)
    3. Fall back to abstract (always works)
    """
    logger.info(f"  Fetching content for {paper_id}...")
    
    content = None
    source = None
    
    # Method 1: Try HTML version
    logger.info(f"  → Trying HTML version...")
    content = extract_from_html_version(paper_id)
    if content:
        source = "html_version"
    else:
        # Method 2: Try PDF extraction
        logger.info(f"  → Trying PDF extraction...")
        content = extract_from_pdf(paper_id)
        if content:
            source = "pdf_extracted"
        else:
            # Method 3: Fall back to abstract
            logger.info(f"  → Using abstract only")
            content = f"{title}\n\n{summary}"
            source = "abstract_only"
    
    return {
        'id': paper_id,
        'title': title,
        'authors': authors,
        'summary': summary,
        'published': published,
        'text': content,
        'content_source': source,
        'text_length': len(content)
    }


def fetch_papers_robust(category: str, max_results: int) -> List[Dict]:
    """
    Fetch papers with robust content extraction.
    """
    logger.info(f"Fetching {max_results} papers from {category}...")
    logger.info("Using robust extraction (HTML → PDF → Abstract)")
    
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    
    for i, result in enumerate(search.results(), 1):
        paper_id = result.entry_id.split('/')[-1]
        
        logger.info(f"[{i}/{max_results}] {result.title[:60]}...")
        
        paper = fetch_paper_with_fallback(
            paper_id=paper_id,
            title=result.title,
            authors=[author.name for author in result.authors[:3]],
            summary=result.summary,
            published=str(result.published.date())
        )
        
        papers.append(paper)
        
        # Be nice to arXiv
        time.sleep(3)
    
    # Log statistics
    html_count = sum(1 for p in papers if p['content_source'] == 'html_version')
    pdf_count = sum(1 for p in papers if p['content_source'] == 'pdf_extracted')
    abstract_count = sum(1 for p in papers if p['content_source'] == 'abstract_only')
    
    logger.info(f"✓ Content sources:")
    logger.info(f"  - HTML version: {html_count}")
    logger.info(f"  - PDF extracted: {pdf_count}")
    logger.info(f"  - Abstract only: {abstract_count}")
    
    return papers

def create_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for text chunks."""
    if not texts:
        return np.array([])
    
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32
    )
    
    logger.info(f"Generated embeddings: {embeddings.shape}")
    return embeddings

def process_all_papers(category: str, num_papers: int, model: SentenceTransformer) -> tuple:
    """Main processing pipeline: fetch, chunk, and embed papers."""
    cache_path = CONFIG['papers_cache_path']
    
    if os.path.exists(cache_path):
        logger.info("Loading papers from cache...")
        with open(cache_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        logger.info(f"Loaded {len(papers)} papers from cache")
    else:
        papers = fetch_papers_robust(category, num_papers)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(papers)} papers to cache")
    
    logger.info("Creating chunks...")
    all_chunks = []
    all_texts = []
    
    for paper in papers:
        text_chunks = split_text_into_chunks(
            paper['text'],
            max_tokens=CONFIG['chunk_max_tokens'],
            overlap=CONFIG['chunk_overlap']
        )
        
        for chunk_idx, chunk_content in enumerate(text_chunks):
            chunk_data = {
                'text': chunk_content,
                'paper_id': paper['id'],
                'paper_title': paper['title'],
                'paper_authors': paper['authors'],
                'published': paper['published'],
                'chunk_index': chunk_idx,
                'total_chunks': len(text_chunks)
            }
            all_chunks.append(chunk_data)
            all_texts.append(chunk_content)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(papers)} papers")
    embeddings = create_embeddings(all_texts, model)
    
    return all_chunks, embeddings


# ============================================================================
# FAISS INDEX
# ============================================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS index from embeddings."""
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot build index with zero embeddings")
    
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index (dimension: {dim})")
    
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    
    logger.info(f"Index built with {index.ntotal} vectors")
    return index


def save_index_and_metadata(index: faiss.Index, chunks: List[Dict]):
    """Save FAISS index and metadata to disk."""
    faiss.write_index(index, CONFIG['faiss_index_path'])
    logger.info(f"Saved index to {CONFIG['faiss_index_path']}")
    
    with open(CONFIG['chunks_metadata_path'], 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata to {CONFIG['chunks_metadata_path']}")


def load_index_and_metadata() -> tuple:
    """Load FAISS index and metadata from disk."""
    if not os.path.exists(CONFIG['faiss_index_path']):
        raise FileNotFoundError(f"Index not found: {CONFIG['faiss_index_path']}")
    
    index = faiss.read_index(CONFIG['faiss_index_path'])
    logger.info(f"Loaded index with {index.ntotal} vectors")
    
    with open(CONFIG['chunks_metadata_path'], 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    return index, chunks


# ============================================================================
# RETRIEVAL
# ============================================================================

def retrieve_relevant_chunks(query: str, model: SentenceTransformer, 
                             index: faiss.Index, chunks: List[Dict], 
                             top_k: int = 3) -> List[Dict]:
    """Retrieve most similar chunks for a query."""
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    
    results = []
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(chunks):
            result = {
                "rank": rank + 1,
                "score": float(distance),
                "text": chunks[idx]["text"],
                "paper_id": chunks[idx]["paper_id"],
                "paper_title": chunks[idx]["paper_title"],
                "paper_authors": chunks[idx]["paper_authors"],
                "published": chunks[idx]["published"],
                "chunk_index": chunks[idx]["chunk_index"],
            }
            results.append(result)
    
    return results


# ============================================================================
# LLM 
# ============================================================================

def build_rag_prompt(query: str, context_chunks: List[Dict]) -> str:
    """Build a prompt for the LLM with retrieved context."""
    context = "\n\n".join([
        f"[Source {i+1}: {chunk['paper_title']}]\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful AI assistant specializing in computer science and natural language processing. You answer questions based on the provided research paper excerpts. Be concise, accurate, and cite the sources when relevant.<|eot_id|><|start_header_id|>user<|end_header_id|>

    Context from research papers:
    {context}

    Question: {query}

    Please answer the question based on the context provided above. If the context doesn't contain enough information, say so.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
    
    return prompt


def generate_answer(query: str, context_chunks: List[Dict], llm: pipeline) -> str:
    """Generate an answer using the LLM with retrieved context."""
    prompt = build_rag_prompt(query, context_chunks)
    
    logger.info("Generating answer with LLM...")
    outputs = llm(
        prompt,
        max_new_tokens=CONFIG['max_new_tokens'],
        temperature=CONFIG['temperature'],
        do_sample=True,
        pad_token_id=llm.tokenizer.eos_token_id
    )
    
    generated_text = outputs[0]['generated_text']
    
    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        answer = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        answer = answer.replace("<|eot_id|>", "").strip()
    else:
        answer = generated_text.split(prompt)[-1].strip()
    
    return answer


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_system():
    """Initialize the complete RAG system."""
    global embedding_model, llm_pipeline, faiss_index, chunks_data
    
    logger.info("="*60)
    logger.info("Initializing Complete RAG System")
    logger.info("="*60)
    
    logger.info(f"Loading embedding model: {CONFIG['embedding_model']}")
    embedding_model = SentenceTransformer(CONFIG['embedding_model'])
    logger.info("✓ Embedding model loaded")
    
    logger.info(f"Loading LLM: {CONFIG['llm_model']}")
    logger.info(f"Device: {CONFIG['device']}")
    logger.info("This may take a few minutes on first run...")
    
    llm_pipeline = pipeline(
        "text-generation",
        model=CONFIG['llm_model'],
        device=CONFIG['device'],
        torch_dtype=torch.float16 if CONFIG['device'] == "cuda" else torch.float32,
    )
    logger.info("✓ LLM loaded")
    
    index_exists = os.path.exists(CONFIG['faiss_index_path'])
    metadata_exists = os.path.exists(CONFIG['chunks_metadata_path'])
    
    #If index and metadata exist, load them; otherwise, build from scratch
    if index_exists and metadata_exists:
        logger.info("Loading existing index...")
        faiss_index, chunks_data = load_index_and_metadata()
    else:
        logger.info("Building new index...")
        chunks_data, embeddings = process_all_papers(
            CONFIG['arxiv_category'],
            CONFIG['num_papers'],
            embedding_model
        )
        faiss_index = build_faiss_index(embeddings)
        save_index_and_metadata(faiss_index, chunks_data)
    
    logger.info("="*60)
    logger.info("✓ RAG System Ready!")
    logger.info(f"  - Indexed: {len(chunks_data)} chunks")
    logger.info(f"  - Papers: {len(set(c['paper_id'] for c in chunks_data))}")
    logger.info(f"  - Device: {CONFIG['device']}")
    logger.info("="*60)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="arXiv RAG System with LLM",
    description="Complete RAG: Retrieval + Generation using Llama 3.2",
    version="4.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    initialize_system()


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the HTML interface."""
    return HTML_TEMPLATE


@app.get("/api")
async def api_root():
    """API root for programmatic access."""
    return {
        "message": "Complete RAG System with LLM Generation",
        "status": "healthy",
        "llm_model": CONFIG['llm_model'],
        "embedding_model": CONFIG['embedding_model'],
        "device": CONFIG['device'],
        "index_size": faiss_index.ntotal if faiss_index else 0,
        "num_chunks": len(chunks_data),
        "num_papers": len(set(c["paper_id"] for c in chunks_data)) if chunks_data else 0,
        "endpoints": {
            "web_ui": "/",
            "search": "/search?q=your+query&k=5",
            "ask": "/ask?q=your+question&k=3"
        }
    }


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(default=3, description="Number of results", ge=1, le=20)
):
    """Retrieval only - returns relevant chunks without generation."""
    try:
        if not embedding_model or not faiss_index:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        results = retrieve_relevant_chunks(
            query=q,
            model=embedding_model,
            index=faiss_index,
            chunks=chunks_data,
            top_k=min(k, len(chunks_data))
        )
        
        return {
            "query": q,
            "num_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ask")
async def ask(
    q: str = Query(..., description="Your question", min_length=1),
    k: int = Query(default=3, description="Number of context chunks", ge=1, le=10)
):
    """Complete RAG - retrieves context and generates answer with LLM."""
    try:
        if not embedding_model or not faiss_index or not llm_pipeline:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        context_chunks = retrieve_relevant_chunks(
            query=q,
            model=embedding_model,
            index=faiss_index,
            chunks=chunks_data,
            top_k=min(k, len(chunks_data))
        )
        
        answer = generate_answer(
            query=q,
            context_chunks=context_chunks,
            llm=llm_pipeline
        )
        
        return {
            "query": q,
            "answer": answer,
            "sources": [
                {
                    "paper_title": chunk["paper_title"],
                    "paper_id": chunk["paper_id"],
                    "authors": chunk["paper_authors"],
                    "published": chunk["published"],
                    "text_preview": chunk["text"][:200] + "..."
                }
                for chunk in context_chunks
            ],
            "num_sources": len(context_chunks)
        }
        
    except Exception as e:
        logger.error(f"Ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """Get system statistics."""
    if not chunks_data:
        return {"message": "No data indexed"}
    
    unique_papers = len(set(chunk["paper_id"] for chunk in chunks_data))
    
    return {
        "total_chunks": len(chunks_data),
        "total_papers": unique_papers,
        "embedding_dimension": faiss_index.d if faiss_index else 0,
        "embedding_model": CONFIG["embedding_model"],
        "llm_model": CONFIG["llm_model"],
        "device": CONFIG["device"],
        "chunk_max_tokens": CONFIG["chunk_max_tokens"],
        "chunk_overlap": CONFIG["chunk_overlap"],
        "arxiv_category": CONFIG["arxiv_category"]
    }


@app.post("/rebuild")
async def rebuild_index():
    """Rebuild index from scratch."""
    try:
        logger.info("Rebuilding index...")
        
        for path in [CONFIG['faiss_index_path'], 
                     CONFIG['chunks_metadata_path'],
                     CONFIG['papers_cache_path']]:
            if os.path.exists(path):
                os.remove(path)
        
        global faiss_index, chunks_data
        chunks_data, embeddings = process_all_papers(
            CONFIG['arxiv_category'],
            CONFIG['num_papers'],
            embedding_model
        )
        faiss_index = build_faiss_index(embeddings)
        save_index_and_metadata(faiss_index, chunks_data)
        
        return {
            "message": "Index rebuilt successfully",
            "total_chunks": len(chunks_data),
            "total_papers": len(set(c["paper_id"] for c in chunks_data))
        }
        
    except Exception as e:
        logger.error(f"Rebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Complete RAG System - Llama 3.2 3B Instruct + Web UI")
    print("="*60)
    print(f"🌐 Web UI: http://127.0.0.1:8000")
    print(f"📚 API Docs: http://127.0.0.1:8000/docs")
    print(f"")
    print(f"Endpoints:")
    print(f"  /       - Simple web interface")
    print(f"  /search - Retrieval only (fast)")
    print(f"  /ask    - Complete RAG with LLM (slower, better)")
    print(f"  /stats  - System information")
    print("="*60 + "\n")
     