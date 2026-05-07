import re
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation and digits, strip whitespace."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)        # remove numbers
    text = re.sub(r'[^\w\s]', '', text)    # remove punctuation
    text = re.sub(r'\s+', ' ', text)       # collapse whitespace
    return text.strip()


def build_vectorizer(max_features: int = 3000) -> TfidfVectorizer:
    """Return a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2)   # unigrams + bigrams for better context
    )


if __name__ == "__main__":
    samples = [
        "WINNER!! Claim your FREE prize now!!!",
        "Hey, are we still on for lunch tomorrow?",
    ]
    for s in samples:
        print(f"Original : {s}")
        print(f"Cleaned  : {clean_text(s)}\n")
