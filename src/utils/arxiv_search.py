from doctest import debug
import os
import time
import html
import json
import random
import requests
import logging
import threading
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    # make sure debug level logs are shown
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])


arxiv_subject = ['', 'physics', 'cs', 'math', 'q-bio'] # subject terms on arxiv
arxiv_search_type = ['abstract', 'all', 'title', 'author', 'comments']

# https://arxiv.org/search/?query=deep+learning&searchtype=all&abstracts=show&order=-announced_date_first&size=200
# https://arxiv.org/search/cs?query=deep+learning&searchtype=all&abstracts=show&order=-announced_date_first&size=200

arxiv_search_url = "https://arxiv.org/search/"

def make_request_with_retries(url: str, headers: dict, timeout: int = 10, retries: int = 3, backoff_factor: float = 0.5):
    """
    Make an HTTP request with random delays and retries with exponential backoff.
    """
    for i in range(retries):
        try:
            # Use a longer, more random delay to appear more human
            time.sleep(random.uniform(3, 10))
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            return response
        except requests.exceptions.RequestException as e:
            if i < retries - 1:
                # Exponential backoff
                sleep_time = backoff_factor * (2 ** i)
                logging.warning(f"Request failed: {e}. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"Request failed after {retries} retries: {e}")
                return None

def build_arxiv_search_url(subject: str = "cs", query: str ="Deep Learning", searchtype: str = "all", abstracts: str ="show", order: str = "-announced_date_first", size: int = 200, arxiv_search_url: str = "https://arxiv.org/search/") -> str:
    """
    Returns an arxiv search URL for retrieving a latest list of items

    Parameters:
    subject (str): Paper subject of interest.
    query (str): User defined search query e.g. paper name
    searchtype (str): Search type  ['abstract', 'all', 'title', 'author', 'comments', ... ]
    abstracts (str): Show abstract or not
    order (str): order to show
    size (int): number of results to show

    Rerurns:
    str: ArXiv Search URL

    Test:
    >>> url = build_arxiv_search_url(size=34)
    """

    # start building the url.
    arxiv_search_url = arxiv_search_url

    # check subject if empty
    if subject.strip():
        arxiv_search_url += subject.strip()

    arxiv_search_url += "?query=" + quote_plus(query.lower().strip()) + "&"
    arxiv_search_url += "searchtype=" + searchtype.lower().strip() + "&"
    arxiv_search_url += "abstracts=" + abstracts.lower().strip() + "&"
    arxiv_search_url += "order=" + order.lower().strip() + "&"

    # control the page sizes
    # arxiv returns 404 if the page sizes are not 25, 50, 100 or 200
    if size <= 25:
        # Any page number <= 25 becomes 25
        size = 25
    elif size > 25 and size <= 50:
        # Any page number between 25 and 50 become 50
        size = 50
    elif size > 50 and size <=100:
        # Any page size between 50 and 100 becomes 100
        size = 100
    else:
        # Any other size always results to 200
        size = 200
    
    # set the size
    arxiv_search_url += "size=" + str(int(size))
    
    # handle if subject is 
    return arxiv_search_url

def scrape_arxiv_url_to_json(url: str, output_file: str = "arxiv_results.json") -> None:
    """
    Scrape arXiv search results from a given URL and save to a JSON file.

    Args:
        url (str): Fully constructed arXiv search URL.
        output_file (str): Filename to save JSON output.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    
    response = make_request_with_retries(url, headers=headers)
    if not response:
        return

    soup = BeautifulSoup(response.text, "html.parser")
    results_list = soup.find("ol", class_="breathe-horizontal")
    results = results_list.find_all("li", class_="arxiv-result") if results_list else []

    papers = []
    for item in results:
        title_tag = item.find("p", class_="title")
        abstract_tag = item.find("span", class_="abstract-full")
        authors_tag = item.find("p", class_="authors")
        link_tag = item.find("p", class_="list-title").find("a")

        paper = {
            "title": title_tag.get_text(strip=True) if title_tag else None,
            "abstract": abstract_tag.get_text(strip=True) if abstract_tag else None,
            "authors": authors_tag.get_text(strip=True) if authors_tag else None,
            "link": link_tag["href"] if link_tag else None
        }
        papers.append(paper)
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_file)
    
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Created directory: {output_dir}")
        except Exception as e:
            logging.error(f"Failed to create directory '{output_dir}': {e}")
            return


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    logging.debug(f"Saved {len(papers)} papers to {output_file}")
    logging.debug(f"Type of output file:{type(output_file)}")

def run_scraper_in_background(url: str, output_file: str = "arxiv_results.json") -> None:
    """
    Run sraper in the background. More efficient and faster
    Same arguments as the scrape_arxiv_url_to_json function.

    Arguments:
    url: str
    output_file

    Return: None

    """
    thread = threading.Thread(target=scrape_arxiv_url_to_json, args=(url, output_file))
    thread.start()
    logging.debug(f"Scraping started in background. Results will be saved to {output_file}.")

def scrape_arxiv_details_from_json(json_file: str) -> list:
    """
    Given a JSON file containing arXiv search results, scrape each paper's abstract page
    and extract detailed metadata.

    Args:
        json_file (str): Path to the JSON file with arXiv search results.

    Returns:
        List[dict]: List of dictionaries with detailed paper metadata.
    """
    if not os.path.exists(json_file):
        logging.error(f"JSON file not found: {json_file}")
        return []

    with open(json_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    detailed_results = []

    for paper in papers:
        # lets get the link property from each paper item o record.
        url = paper.get("link")

        # ignore if there is no url
        if not url:
            continue

        # lets now try and scrape fromt he url property.
        try:
            # get response
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            }
            response = make_request_with_retries(url, headers=headers)

            if not response:
                continue

            # use beautiful soup!
            soup = BeautifulSoup(response.text, "html.parser")

            # get abastract details

            title_tag = soup.find("h1", class_="title")
            abstract_tag = soup.find("blockquote", class_="abstract")
            authors_tag = soup.find("div", class_="authors")
            date_tag = soup.find("div", class_="dateline")


            detailed_results.append({
                "url": url,
                "title": title_tag.get_text(strip=True).replace("Title:", "") if title_tag else None,
                "abstract": abstract_tag.get_text(strip=True).replace("Abstract:", "") if abstract_tag else None,
                "authors": authors_tag.get_text(strip=True).replace("Authors:", "") if authors_tag else None,
                "date": date_tag.get_text(strip=True) if date_tag else None
            })

        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")

    logging.debug(f"Scraped {len(detailed_results)} detailed entries from {json_file}")
    return detailed_results

def scrape_arxiv_abstract_page(url: str) -> dict:
    """Scrape a single arXiv abstract page for metadata."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }
        response = make_request_with_retries(url, headers=headers)
        if not response:
            return {}

        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.find("h1", class_="title")
        abstract_tag = soup.find("blockquote", class_="abstract")
        authors_tag = soup.find("div", class_="authors")
        date_tag = soup.find("div", class_="dateline")

        return {
            "url": url,
            "title": title_tag.get_text(strip=True).replace("Title:", "") if title_tag else None,
            "abstract": abstract_tag.get_text(strip=True).replace("Abstract:", "") if abstract_tag else None,
            "authors": authors_tag.get_text(strip=True).replace("Authors:", "") if authors_tag else None,
            "date": date_tag.get_text(strip=True) if date_tag else None
        }

    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return {}

def scrape_arxiv_details_from_json_threaded(json_file: str, max_workers: int = 3) -> list:
    """Scrape arXiv abstract pages concurrently using threads."""
    if not os.path.exists(json_file):
        logging.error(f"JSON file not found: {json_file}")
        return []

    with open(json_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    urls = [paper.get("link") for paper in papers if paper.get("link")]
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(scrape_arxiv_abstract_page, url): url for url in urls}
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                results.append(result)

    logging.debug(f"Scraped {len(results)} detailed entries using threads.")
    return results

def save_arxiv_scraped_details(results: list, output_file: str = "clean/arxiv_clean.json") -> None:
    """
    Save scraped arXiv details to a JSON file in a cleaned directory.

    Args:
        results (list): List of scraped paper metadata.
        output_file (str): Path to output JSON file.
    """
    logging.debug(f"First item in list: {results[0]}")
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Created cleaned directory: {output_dir}")
        except Exception as e:
            logging.error(f"Failed to create directory '{output_dir}': {e}")
            return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.debug(f"Saved cleaned results to {output_file}")

def download_pdf(arxiv_id: str, save_dir: str = "data/pdfs/arxiv") -> None:
    """
    Definition: Download a paper using arxiv id

    Arguments:
        arxiv_id: str -> arxiv_id
        save_dir: str -> file location
    
    Return: None

    >>> download_pdf("2510.26641") # attention is all you need paper example

    """
    logging.debug(f"Save dir:{save_dir}")
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    logging.debug(f"arXiv url: {pdf_url}")

    # file path
    save_path = os.path.join(save_dir, f"{arxiv_id}.pdf")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }
        logging.debug(f"Downloading {pdf_url} to {save_path}")
        response = make_request_with_retries(pdf_url, headers=headers)

        if response:
            with open(save_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Saved: {save_path}")

    # Exception
    except Exception as e:
        logging.error(f"Failed to download {pdf_url}: {e}")

def get_pdf_arxiv(cleaned_json: str = "clean/arxiv_clean.json", save_dir: str = "data/pdfs/arxiv") -> None:
    """
    Description:
        Retrieves each paper's url from the cleaned json file
        Extracts the paper id from the url
        Passes the paper id to the download function to get the pdf
    
    Argument: cleaned_json file path.

    Dependencies:
        download_pdf function

    >>> safe_pdf_arxiv()

    """

    os.makedirs(save_dir, exist_ok=True)

    logging.debug("Threadsafe pdf arxiv")
    logging.info(f"Saving to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    try:
        with open(cleaned_json, "r", encoding="utf-8") as f:
            papers = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read JSON file: {e}")
        return

    for paper in papers:
        url = paper.get("url", "")
        if not url.startswith("https://arxiv.org/abs/"):
            logging.warning(f"Skipping invalid URL: {url}")
            continue

        # get the id form the url then pass it to the download pdf function.
        arxiv_id = url.split("/")[-1]
        thread = threading.Thread(target=download_pdf, args=(arxiv_id, save_dir), daemon=True)
        thread.start()
        logging.debug(f"Started thread for {arxiv_id}")
