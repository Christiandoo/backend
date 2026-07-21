import re
import nltk

# Download stopwords sekali saat startup
def download_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

download_nltk_resources()

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("indonesian"))


def clean_text(text: str) -> str:
    """
    Langkah 1: Cleaning
    - Lowercase
    - Hapus URL
    - Hapus mention & hashtag
    - Hapus angka
    - Hapus tanda baca & karakter spesial
    - Hapus spasi berlebih
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # hapus URL
    text = re.sub(r"@\w+|#\w+", "", text)            # hapus mention & hashtag
    text = re.sub(r"\d+", "", text)                  # hapus angka
    text = re.sub(r"[^\w\s]", " ", text)             # hapus tanda baca
    text = re.sub(r"_+", " ", text)                  # hapus underscore
    text = re.sub(r"\s+", " ", text).strip()         # hapus spasi berlebih
    return text


def remove_stopwords(text: str) -> str:
    """
    Langkah 2: Stopword Removal (Bahasa Indonesia)
    Digunakan HANYA untuk pipeline SVM, bukan IndoBERT.
    """
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def preprocess_for_svm(text: str) -> tuple[str, str]:
    """
    Pipeline untuk SVM:
    Input → cleaning → stopword removal
    Return: (after_cleaning, after_stopword)
    """
    cleaned = clean_text(text)
    no_stopword = remove_stopwords(cleaned)
    return cleaned, no_stopword


def preprocess_for_indobert(text: str) -> str:
    """
    Pipeline untuk IndoBERT:
    Input → cleaning saja (TANPA stopword removal)
    IndoBERT butuh konteks kalimat yang lebih natural.
    """
    return clean_text(text)
