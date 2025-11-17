import string
import re

def text_cleanser(text: str, case_sensitive: bool = False)-> str:
    """
    Applies a series of cleansing steps to a raw text string.

    Args:
        text(str): The raw input text
        case_sensitive (bool): Optionally skip lowercasing.

    """

    # Step 1. Case sensitivity.
    if not case_sensitive:
        text = text.lower()
    else:
        ...
    
    # Step 2. Removing HTML tags.
    text = re.sub('<.*?>','',text)

    # Step 3. Remove URLS
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    if case_sensitive:
        ...
    else:
        text = text.translate(str.maketrans('','',string.punctuation))
    
    # Normalize and remove extra space
    text = re.sub(r'\s+', ' ', text).strip()

    return text