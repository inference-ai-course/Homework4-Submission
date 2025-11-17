# Task 1: Data Collection: Obtain 50 arXiv cs.CL PDFs 

import requests
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET

SUBCATEGORY = "cs.CL"
MAX_PAPERS = 50
OUTPUT_FILE = "arxiv.json"
PDF_DIR = Path("pdfs")
PDF_DIR.mkdir(exist_ok=True)

ARXIV_API = (
    "http://export.arxiv.org/api/query"
    f"?search_query=cat:{SUBCATEGORY}"
    f"&start=0&max_results={MAX_PAPERS}"
    "&sortBy=submittedDate&sortOrder=descending"
)

# Helpers

def get_text(node, default=""):
    """Safely extract node.text or return default."""
    if node is None:
        return default
    text = node.text
    return text.strip().replace("\n", " ") if text else default

# Parse ArXiv entry safely

def parse_entry(entry):
    # ID
    id_node = entry.find("{http://www.w3.org/2005/Atom}id")
    paper_id = get_text(id_node).split("/")[-1]

    # Title
    title_node = entry.find("{http://www.w3.org/2005/Atom}title")
    title = get_text(title_node)

    # Abstract
    abs_node = entry.find("{http://www.w3.org/2005/Atom}summary")
    abstract = get_text(abs_node)

    # Authors
    authors = []
    for a in entry.findall("{http://www.w3.org/2005/Atom}author"):
        name_node = a.find("{http://www.w3.org/2005/Atom}name")
        authors.append(get_text(name_node))

    # Date
    pub_node = entry.find("{http://www.w3.org/2005/Atom}published")
    date = get_text(pub_node)

    # PDF link
    pdf_url = None
    for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href")

    return {
        "id": paper_id,
        "title": title or "(missing title)",
        "authors": authors,
        "date": date,
        "abstract": abstract or "(missing abstract)",
        "pdf_url": pdf_url
    }

# Download PDF

def download_pdf(paper_id, pdf_url):
    if not pdf_url:
        return None  # cannot download if link missing

    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    if pdf_path.exists():
        return str(pdf_path)

    print(f"   → Downloading PDF: {pdf_url}")
    response = requests.get(pdf_url, stream=True)
    if not response.ok:
        print(f"     PDF download failed: {response.status_code}")
        return None

    with open(pdf_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return str(pdf_path)

# Main
def fetch_arxiv_papers():
    #print("Fetching metadata from arXiv API...")
    response = requests.get(ARXIV_API)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")

    papers = []
    for i, entry in enumerate(entries):
        #print(f"[{i+1}/{len(entries)}] Processing metadata...")
        paper = parse_entry(entry)

        # Try downloading PDF (may be missing)
        paper["pdf_path"] = download_pdf(paper["id"], paper["pdf_url"])

        papers.append(paper)
        time.sleep(0.5)

    return papers


def save_json(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n Saved {len(data)} papers to {OUTPUT_FILE}")
    print(f" PDFs stored in: {PDF_DIR.absolute()}")


if __name__ == "__main__":
    papers = fetch_arxiv_papers()
    save_json(papers)

# Task 2: Text Extraction (PDF → Text)

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.
    """
    if pdf_path is None:
        return ""
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Could not open PDF {pdf_path}: {e}")
        return ""
    
    pages = []
    for page in doc:
        page_text = page.get_text()  # get raw text from page
        # (Optional) clean page_text here (remove headers/footers)
        if page_text:
            # Clean multiple spaces/newlines
            cleaned = " ".join(page_text.split())
            pages.append(cleaned)

    doc.close()
        
    full_text = "\n".join(pages)
    return full_text

papers = fetch_arxiv_papers()

for paper in papers:
    pdf_path = paper.get("pdf_path")

    print(f"Extracting text from {paper['id']}...")
    full_text = extract_text_from_pdf(pdf_path)

    paper["pdf_text"] = full_text