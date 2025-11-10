# Get 50 PDF research papers from arxiv and save them.
# What will i use to get the papers from arxiv
# where will i save the papers in an organixed way

import os
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()          # Also log to console
    ]
)

DATA_DIR = "data"
PDF_FILES_DIR = "pdf-files"


def prepare_folders(data_dir: str = "data", files_dir: str = "data/pdf-files"):
    """ Prepare necessary folders: Holding data """
    
    a = os.makedirs(data_dir, exist_ok=True)
    b = os.makedirs(files_dir, exist_ok=True)
    logging.info(f"Directories created!\n{a}\n{b}")

