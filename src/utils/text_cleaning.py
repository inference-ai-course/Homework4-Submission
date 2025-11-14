import re
import string

def clean_text_pipeline(text: str, case_sensitive: bool = False)-> str:
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

raw_text_general = "HELLO World! Check out my site: https://example.com. It's awesome!"
cleaned_general = clean_text_pipeline(raw_text_general, case_sensitive=True)
print("\n--- General Text Output ---")
print(f"Raw:     {raw_text_general}")
print(f"Cleaned: {cleaned_general}")

import spacy

nlp = spacy.load("en_core_web_sm")
print("Pipeline:", nlp.pipe_names)
doc = nlp("I was reading the paper.")
token = doc[0]  # 'I'
print(token.morph)  # 'Case=Nom|Number=Sing|Person=1|PronType=Prs'
print(token.morph.get("PronType"))  # ['Prs']