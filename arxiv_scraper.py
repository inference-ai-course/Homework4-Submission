"""
arXiv Paper Abstract Scraper for Finance/Quant Research
Extracts papers from arXiv categories, cleans HTML with Trafilatura, and uses OCR (optional)
"""

import requests
from bs4 import BeautifulSoup
import trafilatura
import json
import time
from datetime import datetime
import os
import re

# Optional imports for OCR
try:
    import pytesseract
    from PIL import Image
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from io import BytesIO
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("OCR libraries not available. Running in Trafilatura-only mode.")

class ArxivScraper:
    def __init__(self, category="q-fin", max_papers=200):
        """
        Initialize scraper for arXiv papers
        
        Args:
            category: arXiv category (e.g., 'q-fin' for quantitative finance, 
                     'cs.CL' for computation and language)
            max_papers: Maximum number of papers to scrape
        """
        self.category = category
        self.max_papers = max_papers
        self.base_url = "https://arxiv.org"
        self.papers = []
        
    def get_paper_list(self):
        """Fetch list of recent papers from arXiv category using /list/ endpoint"""
        paper_urls = []
        start = 0
        papers_per_page = 50  # arXiv limit - don't exceed this
        
        print(f"Fetching papers from category: {self.category}")
        
        while len(paper_urls) < self.max_papers:
            # Use the /list/ endpoint which shows recent papers
            list_url = f"{self.base_url}/list/{self.category}/recent?skip={start}&show={papers_per_page}"
            
            print(f"Fetching from: {list_url}")
            try:
                response = requests.get(list_url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all paper links with title='Abstract'
                links = soup.find_all('a', title='Abstract')
                
                if not links:
                    print("No more papers found")
                    break
                
                for link in links:
                    if len(paper_urls) >= self.max_papers:
                        break
                    
                    href = link.get('href')
                    if href:
                        # Extract paper ID from href (e.g., /abs/2401.12345)
                        paper_id = href.split('/abs/')[-1]
                        full_url = f"{self.base_url}/abs/{paper_id}"
                        paper_urls.append(full_url)
                
                print(f"Collected {len(paper_urls)} paper URLs so far...")
                
                if len(links) < papers_per_page:
                    # No more papers available
                    break
                
                start += papers_per_page
                time.sleep(1)  # Be respectful to arXiv servers
                
            except Exception as e:
                print(f"Error fetching paper list: {str(e)}")
                break
        
        return paper_urls[:self.max_papers]
    
    def scrape_paper_trafilatura(self, url):
        """Scrape paper abstract page using Trafilatura for cleaning"""
        
        try:
            # Download HTML
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Use Trafilatura for content extraction
            extracted = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
                output_format='json'
            )
            
            # Also parse with BeautifulSoup for structured data
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1', class_='title')
            if not title_elem:
                title_elem = soup.find('h1', class_='title mathjax')
            title = title_elem.text.replace('Title:', '').strip() if title_elem else "N/A"
            
            # Extract authors
            authors_elem = soup.find('div', class_='authors')
            authors = []
            if authors_elem:
                author_links = authors_elem.find_all('a')
                authors = [a.text.strip() for a in author_links]
            
            # Extract abstract
            abstract_elem = soup.find('blockquote', class_='abstract')
            if not abstract_elem:
                abstract_elem = soup.find('blockquote', class_='abstract mathjax')
            abstract = ""
            if abstract_elem:
                abstract = abstract_elem.text.replace('Abstract:', '').strip()
            
            # Extract submission date
            dateline = soup.find('div', class_='dateline')
            date = dateline.text.strip() if dateline else "N/A"
            
            # Extract categories/subjects
            subjects = soup.find('span', class_='primary-subject')
            categories = subjects.text.strip() if subjects else self.category
            
            # Extract arXiv ID from URL
            arxiv_id = url.split('/abs/')[-1]
            
            paper_data = {
                'url': url,
                'paper_id': arxiv_id,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'date': date,
                'categories': categories,
                'scraped_at': datetime.now().isoformat()
            }
            
            return paper_data
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
    
    def scrape_paper_with_ocr(self, url, use_selenium=False):
        """
        Scrape paper and use OCR on screenshot as backup/enhancement
        
        Args:
            url: Paper URL
            use_selenium: If True, use Selenium for screenshot (more reliable but slower)
        """
        if not OCR_AVAILABLE:
            print("OCR not available. Using Trafilatura only.")
            return self.scrape_paper_trafilatura(url)
            
        # First try with Trafilatura
        paper_data = self.scrape_paper_trafilatura(url)
        
        if not paper_data or not paper_data['abstract']:
            print(f"Trafilatura failed for {url}, trying OCR...")
            
            if use_selenium:
                paper_id = url.split('/abs/')[-1]
                paper_data = self._ocr_with_selenium(url, paper_id)
            else:
                print("OCR requires Selenium. Set use_selenium=True")
        
        return paper_data
    
    def _ocr_with_selenium(self, url, paper_id):
        """Use Selenium to take screenshot and apply OCR"""
        if not OCR_AVAILABLE:
            return None
            
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            
            # Wait for page to load
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "abstract")))
            
            # Take screenshot
            screenshot = driver.get_screenshot_as_png()
            img = Image.open(BytesIO(screenshot))
            
            # Apply OCR
            ocr_text = pytesseract.image_to_string(img)
            
            # Try to extract structured info from OCR text
            lines = ocr_text.split('\n')
            title = ""
            abstract = ""
            authors = []
            
            # Simple parsing logic (can be improved)
            in_abstract = False
            for i, line in enumerate(lines):
                line = line.strip()
                if 'Title:' in line or (i < 5 and len(line) > 20):
                    title = line.replace('Title:', '').strip()
                elif 'Authors:' in line:
                    authors_text = line.replace('Authors:', '').strip()
                    authors = [a.strip() for a in authors_text.split(',')]
                elif 'Abstract' in line or 'abstract' in line.lower():
                    in_abstract = True
                elif in_abstract and line:
                    abstract += line + " "
            
            driver.quit()
            
            paper_data = {
                'url': url,
                'paper_id': paper_id,
                'title': title,
                'abstract': abstract.strip(),
                'authors': authors,
                'date': 'N/A',
                'scraped_at': datetime.now().isoformat(),
                'method': 'OCR'
            }
            
            return paper_data
            
        except Exception as e:
            print(f"OCR failed for {url}: {str(e)}")
            return None
    
    def scrape_all(self, use_ocr=False):
        """
        Main method to scrape all papers
        
        Args:
            use_ocr: Whether to use OCR as backup method
        """
        print(f"Starting arXiv scraper for category: {self.category}")
        paper_urls = self.get_paper_list()
        print(f"\nFound {len(paper_urls)} papers to scrape\n")
        
        for i, url in enumerate(paper_urls):
            print(f"Scraping paper {i+1}/{len(paper_urls)}: {url}")
            
            if use_ocr and OCR_AVAILABLE:
                paper_data = self.scrape_paper_with_ocr(url, use_selenium=True)
            else:
                paper_data = self.scrape_paper_trafilatura(url)
            
            if paper_data:
                self.papers.append(paper_data)
            
            # Be respectful to arXiv servers
            time.sleep(1)
        
        print(f"\nSuccessfully scraped {len(self.papers)} papers")
        return self.papers
    
    def save_to_json(self, filename="arxiv_clean.json"):
        """Save scraped papers to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.papers, f, indent=2, ensure_ascii=False)
        
        # Check file size
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f"\nSaved {len(self.papers)} papers to {filename}")
        print(f"File size: {file_size:.2f} MB")
        
        if file_size > 1:
            print("⚠️  WARNING: File size exceeds 1MB")
            print("Consider reducing max_papers or filtering fields")


def main():
    """
    Example usage for finance/quant categories
    
    Popular arXiv categories for finance/quant:
    - q-fin: All quantitative finance
    - q-fin.CP: Computational Finance
    - q-fin.PM: Portfolio Management
    - q-fin.PR: Pricing of Securities
    - q-fin.RM: Risk Management
    - q-fin.ST: Statistical Finance
    - q-fin.TR: Trading and Market Microstructure
    - stat.AP: Statistics - Applications (often includes finance)
    - cs.CE: Computational Engineering, Finance, and Science
    
    Other popular categories:
    - cs.CL: Computation and Language (NLP)
    - cs.LG: Machine Learning
    - cs.AI: Artificial Intelligence
    """
    
    # Choose your category - CHANGE THIS to your preferred category
    category = "cs.CL"  # Statistical Finance
    
    print("=" * 60)
    print(f"arXiv Paper Scraper")
    print(f"Category: {category}")
    print(f"OCR Available: {OCR_AVAILABLE}")
    print("=" * 60)
    print()
    
    # Initialize scraper
    scraper = ArxivScraper(category=category, max_papers=200)
    
    # Scrape papers (use_ocr=False for faster scraping without OCR)
    papers = scraper.scrape_all(use_ocr=False)
    
    # Save to JSON
    scraper.save_to_json("arxiv_clean.json")
    
    # Print sample
    if papers:
        print("\n" + "=" * 60)
        print("Sample Paper:")
        print("=" * 60)
        print(f"Title: {papers[0]['title']}")
        print(f"Authors: {', '.join(papers[0]['authors'][:3])}")
        print(f"Date: {papers[0]['date']}")
        print(f"Abstract Preview: {papers[0]['abstract'][:300]}...")
        print("=" * 60)
        print(f"\n✅ Scraping completed! Check arxiv_clean.json for all {len(papers)} papers.")


if __name__ == "__main__":
    main()