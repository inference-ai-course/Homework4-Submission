import pytesseract
from pdf2image import convert_from_path
import re
import os
import logging
from PIL import Image

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path, dpi=300, lang="eng", max_pages=None, start_page=1, last_page=1, output_folder="converted"):
    """
    Convert PDF pages to images and extract text using Tesseract OCR.

    Args:
        pdf_path (str): Path to the PDF file.
        dpi (int): Resolution for image conversion.
        lang (str): Language for OCR.
        max_pages (int or None): Max number of pages to process.

    Returns:
        str: Cleaned OCR-extracted text.
    """

    # output_folder
    if isinstance(output_folder, str) and output_folder:
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder=None


    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = convert_from_path(pdf_path, dpi=dpi)
    logging.debug(f"Number of pages: {pages}")
    if max_pages is not None:
        pages = pages[:max_pages]
        logging.debug("")

    text_chunks = [pytesseract.image_to_string(page, lang=lang) for page in pages]
    return re.sub(r"\s+", " ", " ".join(text_chunks)).strip()

def ensure_dir(path):
    """Make sure the path exists. """
    os.makedirs(path, exist_ok=True)

def convert_pdf_to_images(pdf_path, output_dir = "pdf_image", dpi=300):
    ensure_dir(output_dir)
    pages = convert_from_path(pdf_path=pdf_path, dpi=dpi)
    image_paths = []

    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f"page_{i}.png")
        page.save(image_path, "PNG")
        image_paths.append(image_path)
    
    return image_paths

def run_tesseract_on_images(image_paths, lang="eng"):
    text_chunks = []

    for path in image_paths:
        image = Image.open(path)
        ocr_text = pytesseract.image_to_string(image, lang=lang)
        text_chunks.append(ocr_text)
    
    return re.sub(r"\s+", " ", " ".join(text_chunks)).strip()

def save_text_to_file(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_pdf_to_text(pdf_path, image_dir="pdf_image", text_output="output.txt", dpi=300, lang="eng"):
    """Combines the pipeline PDF > Image > OCR > text file"""
    image_paths = convert_pdf_to_images(pdf_path, output_dir=image_dir, dpi=dpi)
    text = run_tesseract_on_images(image_paths, lang=lang)
    save_text_to_file(text, text_output)
    return text_output

def batch_process_pdfs(pdfs, converted_pdfs, image_dir="pdf_image", dpi=300, lang="eng"):
    """
    Batch process multiple PDFs: convert to images, extract text, and save to text files.

    Args:
        pdfs (list): List of input PDF file paths.
        converted_pdfs (list): List of output text file paths.
        image_dir (str): Directory to store intermediate images.
        dpi (int): Resolution for image conversion.
        lang (str): Language for OCR.

    Returns:
        list: List of output text file paths successfully processed.
    """
    if len(pdfs) != len(converted_pdfs):
        raise ValueError("pdfs and converted_pdfs must be the same length")

    results = []
    for pdf_path, text_output in zip(pdfs, converted_pdfs):
        try:
            image_subdir = os.path.join(image_dir, os.path.splitext(os.path.basename(pdf_path))[0])
            image_paths = convert_pdf_to_images(pdf_path, output_dir=image_subdir, dpi=dpi)
            text = run_tesseract_on_images(image_paths, lang=lang)
            save_text_to_file(text, text_output)
            results.append(text_output)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    return results