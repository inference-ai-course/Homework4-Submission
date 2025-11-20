"""
arXiv PDF Downloader
Downloads PDFs for papers scraped from arXiv
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Dict


class ArxivPDFDownloader:
    def __init__(self, output_dir: str = "arxiv_pdfs"):
        """
        Initialize PDF downloader
        
        Args:
            output_dir: Directory to save downloaded PDFs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.base_url = "https://arxiv.org/pdf/"
    
    def download_pdf(self, paper_id: str, max_retries: int = 3) -> bool:
        """
        Download a single PDF from arXiv
        
        Args:
            paper_id: arXiv paper ID (e.g., "2401.12345")
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        # Clean paper_id (remove any version info like v1, v2)
        clean_id = paper_id.split('v')[0]
        
        pdf_url = f"{self.base_url}{clean_id}.pdf"
        output_path = self.output_dir / f"{clean_id}.pdf"
        
        # Skip if already exists
        if output_path.exists():
            print(f"  ⏭️  Already exists: {clean_id}.pdf")
            return True
        
        # Download with retries
        for attempt in range(max_retries):
            try:
                print(f"  📥 Downloading: {clean_id}.pdf")
                response = requests.get(pdf_url, timeout=30, stream=True)
                response.raise_for_status()
                
                # Save PDF
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify file size
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                print(f"  ✅ Downloaded: {clean_id}.pdf ({file_size:.2f} MB)")
                return True
            
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"  ⚠️  Failed to download {clean_id}.pdf after {max_retries} attempts")
                    return False
        
        return False
    
    def download_from_json(self, json_file: str = "arxiv_clean.json", 
                          max_papers: int = 50) -> Dict:
        """
        Download PDFs for papers listed in JSON file
        
        Args:
            json_file: Path to JSON file from arxiv_scraper
            max_papers: Maximum number of papers to download
            
        Returns:
            Dictionary with download statistics
        """
        # Load paper metadata
        with open(json_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        print(f"Found {len(papers)} papers in {json_file}")
        papers_to_download = papers[:max_papers]
        print(f"Will download {len(papers_to_download)} papers\n")
        
        # Download PDFs
        successful = 0
        failed = []
        
        for i, paper in enumerate(papers_to_download):
            paper_id = paper.get('paper_id') or paper.get('url', '').split('/abs/')[-1]
            print(f"[{i+1}/{len(papers_to_download)}] {paper_id}")
            
            if self.download_pdf(paper_id):
                successful += 1
            else:
                failed.append(paper_id)
            
            # Be respectful to arXiv servers
            time.sleep(1)
            print()
        
        # Summary
        stats = {
            'total_attempted': len(papers_to_download),
            'successful': successful,
            'failed': len(failed),
            'failed_ids': failed,
            'output_directory': str(self.output_dir)
        }
        
        return stats
    
    def download_from_list(self, paper_ids: List[str]) -> Dict:
        """
        Download PDFs from a list of paper IDs
        
        Args:
            paper_ids: List of arXiv paper IDs
            
        Returns:
            Dictionary with download statistics
        """
        print(f"Downloading {len(paper_ids)} papers\n")
        
        successful = 0
        failed = []
        
        for i, paper_id in enumerate(paper_ids):
            print(f"[{i+1}/{len(paper_ids)}] {paper_id}")
            
            if self.download_pdf(paper_id):
                successful += 1
            else:
                failed.append(paper_id)
            
            time.sleep(1)
            print()
        
        stats = {
            'total_attempted': len(paper_ids),
            'successful': successful,
            'failed': len(failed),
            'failed_ids': failed,
            'output_directory': str(self.output_dir)
        }
        
        return stats


def main():
    """Example usage"""
    print("=" * 60)
    print("arXiv PDF Downloader")
    print("=" * 60)
    print()
    
    # Initialize downloader
    downloader = ArxivPDFDownloader(output_dir="arxiv_pdfs")
    
    # Option 1: Download from arxiv_clean.json
    try:
        stats = downloader.download_from_json("arxiv_clean.json", max_papers=50)
        
        print("\n" + "=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"Total attempted: {stats['total_attempted']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Output directory: {stats['output_directory']}")
        
        if stats['failed_ids']:
            print(f"\nFailed paper IDs:")
            for paper_id in stats['failed_ids']:
                print(f"  - {paper_id}")
        
        print("\n✅ Next step: Run pdf_processor.py to extract text")
    
    except FileNotFoundError:
        print("❌ arxiv_clean.json not found!")
        print("\nPlease run arxiv_scraper.py first to get paper metadata")
        print("Or provide a list of paper IDs manually")
        
        # Option 2: Download from manual list (example)
        # example_ids = ["2401.12345", "2401.12346", "2401.12347"]
        # stats = downloader.download_from_list(example_ids)


if __name__ == "__main__":
    main()