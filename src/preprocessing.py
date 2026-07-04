"""
preprocessing.py
-----------------
Handles all text-cleaning and normalisation steps required before an email
message can be fed into a machine-learning model.

Pipeline (in order):
    1. Lowercase conversion       -> normalises case so "FREE" == "free"
    2. Removing punctuation       -> strips symbols that add noise
    3. Removing numbers           -> digits rarely help distinguish spam/ham
    4. Tokenization                -> splits text into individual words
    5. Removing stopwords         -> drops common words ("the", "is", "and")
    6. Stemming                   -> reduces words to their root form
    7. Lemmatization               -> reduces words to their dictionary form
    8. Re-joining into cleaned text

Both stemming and lemmatization are implemented; the pipeline exposes both so
you can compare which works best for this dataset. By default we lemmatize
(gives more readable / linguistically valid tokens), but a stemmed version is
also available via `clean_text(..., method="stem")`.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure required NLTK resources are available (safe to call repeatedly).
for resource in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


def to_lowercase(text: str) -> str:
    """Step 1: Convert all characters to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove URLs but keep a placeholder token 'urllink' -- the *presence*
    of a link is itself a strong spam signal, so we don't want to lose that
    information entirely, we just remove the noisy raw URL text."""
    return re.sub(r"http\S+|www\.\S+", " urllink ", text)


def remove_punctuation(text: str) -> str:
    """Step 2: Remove punctuation marks (keeps words clean for tokenizing)."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Step 3: Remove standalone digits/numbers."""
    return re.sub(r"\d+", "", text)


def tokenize(text: str):
    """Step 4: Split text into a list of word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens):
    """Step 5: Remove common English stopwords that carry little meaning."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def stem_tokens(tokens):
    """Step 6 (alt): Reduce words to their root/stem form, e.g. 'running' -> 'run'."""
    return [STEMMER.stem(t) for t in tokens]


def lemmatize_tokens(tokens):
    """Step 7: Reduce words to their dictionary/base form, e.g. 'better' -> 'good'."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def clean_text(text: str, method: str = "lemma") -> str:
    """
    Full preprocessing pipeline applied to a single raw email string.

    Parameters
    ----------
    text : str
        Raw email text.
    method : str
        "lemma" (default) to lemmatize, or "stem" to stem instead.

    Returns
    -------
    str
        Cleaned, normalised text ready for feature extraction.
    """
    if not isinstance(text, str):
        return ""

    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)

    if method == "stem":
        tokens = stem_tokens(tokens)
    else:
        tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


if __name__ == "__main__":
    sample = "CONGRATULATIONS!!! You have WON $1,000,000! Click http://scam.com NOW!!!"
    print("Original :", sample)
    print("Lemmatized:", clean_text(sample, method="lemma"))
    print("Stemmed   :", clean_text(sample, method="stem"))
