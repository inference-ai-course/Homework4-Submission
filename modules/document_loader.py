"""
Document Loader Module
Handles file upload, validation, and text extraction from PDF files.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import pdfplumber
from loguru import logger


class DocumentLoader:
    """Handles PDF file validation and text extraction."""
    
    def __init__(self, max_file_size_mb: int = 10, upload_directory: str = "./data/uploads"):
        """
        Initialize DocumentLoader.
        
        Args:
            max_file_size_mb: Maximum allowed file size in MB
            upload_directory: Directory to store uploaded files
        """
        self.max_file_size_mb = max_file_size_mb
        self.upload_directory = Path(upload_directory)
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"DocumentLoader initialized. Upload directory: {self.upload_directory}")
        
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate file format and size.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return False, "File does not exist"
            
            # Check file extension
            if file_path.suffix.lower() != '.pdf':
                return False, f"Invalid file format. Expected .pdf, got {file_path.suffix}"
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return False, f"File size ({file_size_mb:.2f}MB) exceeds limit ({self.max_file_size_mb}MB)"
            
            logger.info(f"File validation passed: {file_path.name} ({file_size_mb:.2f}MB)")
            return True, "File is valid"
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return False, f"Error validating file: {str(e)}"
    
    def extract_text(self, file_path: str) -> Tuple[Optional[str], dict, str]:
        """
        Extract text from PDF file using pdfplumber.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata, message)
        """
        try:
            # Validate file first
            is_valid, message = self.validate_file(file_path)
            if not is_valid:
                return None, {}, message
            
            file_path = Path(file_path)
            extracted_text = ""
            page_texts = []
            
            # Extract text using pdfplumber
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        page_texts.append({
                            'page_num': page_num,
                            'text': page_text
                        })
                        extracted_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                    
                    logger.debug(f"Extracted text from page {page_num}/{total_pages}")
            
            # Clean up extracted text
            extracted_text = self._clean_text(extracted_text)
            
            # Calculate metadata
            metadata = {
                'filename': file_path.name,
                'total_pages': total_pages,
                'total_characters': len(extracted_text),
                'estimated_tokens': len(extracted_text) // 4,  # Rough estimate: 1 token ≈ 4 chars
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"Successfully extracted text: {metadata['total_pages']} pages, "
                       f"{metadata['estimated_tokens']} tokens")
            
            return extracted_text, metadata, "Text extraction successful"
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None, {}, f"Error extracting text: {str(e)}"
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing artifacts and normalizing spacing.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        import re
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def save_uploaded_file(self, uploaded_file) -> Tuple[Optional[str], str]:
        """
        Save uploaded file from Gradio to upload directory.
        
        Args:
            uploaded_file: File object from Gradio
            
        Returns:
            Tuple of (saved_file_path, message)
        """
        try:
            if uploaded_file is None:
                return None, "No file uploaded"
            
            # Get the file path from Gradio's temporary storage
            if hasattr(uploaded_file, 'name'):
                temp_path = uploaded_file.name
            else:
                temp_path = uploaded_file
            
            # Create a unique filename
            original_filename = Path(temp_path).name
            save_path = self.upload_directory / original_filename
            
            # If file already exists, add a number suffix
            counter = 1
            while save_path.exists():
                stem = Path(original_filename).stem
                suffix = Path(original_filename).suffix
                save_path = self.upload_directory / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy file to upload directory (Gradio already saved it to temp location)
            import shutil
            shutil.copy2(temp_path, save_path)
            
            logger.info(f"File saved to: {save_path}")
            return str(save_path), "File saved successfully"
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            return None, f"Error saving file: {str(e)}"
