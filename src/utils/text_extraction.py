import pymupdf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# start by extracting text from pdfs

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Sequential version: Extract text from PDF page by page.
    """
    doc = pymupdf.open(pdf_path) # open a document
    out = open(f"{pdf_path}.txt", "wb") # create a text output
    extracted_text = []

    for page in doc: # iterate the document pages
        text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
        out.write(text) # write text of page
        out.write(bytes((12,))) # write page delimiter (form feed 0x0C)
        extracted_text.append(text.decode("utf8"))
    full_text = "\n".join(extracted_text)
    out.close()
    return full_text


def extract_page_text(doc: pymupdf.Document, page_num: int) -> Tuple[int, bytes]:
    """
    Helper function to extract text from a single page.
    Returns tuple of (page_number, text) to maintain order.
    """
    page = doc[page_num]
    text = page.get_text().encode("utf8")
    return (page_num, text)


def extract_text_from_pdf_threaded(pdf_path: str, max_workers: int = 4) -> str:
    """
    Threaded version: Extract text from PDF using multiple threads.
    
    Args:
        pdf_path: Path to the PDF file
        max_workers: Maximum number of worker threads (default: 4)
    
    Returns:
        Full extracted text as a string

    Example:
        >>> # Threaded version (faster for large PDFs)
        >>> extract_text_from_pdf_threaded("src/data/downloads/pdf/2511.04886.pdf", max_workers=4)
    """
    doc = pymupdf.open(pdf_path)
    out = open(f"{pdf_path}.txt", "wb")
    
    # Dictionary to store results with page numbers as keys
    page_texts = {}
    
    # Use ThreadPoolExecutor to process pages concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pages for processing
        future_to_page = {
            executor.submit(extract_page_text, doc, page_num): page_num 
            for page_num in range(len(doc))
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_page):
            page_num, text = future.result()
            page_texts[page_num] = text
    
    # Write pages in order to output file
    extracted_text = []
    for page_num in sorted(page_texts.keys()):
        text = page_texts[page_num]
        out.write(text)
        out.write(bytes((12,)))  # write page delimiter (form feed 0x0C)
        extracted_text.append(text.decode("utf8"))
    
    full_text = "\n".join(extracted_text)
    out.close()
    doc.close()
    
    return full_text
