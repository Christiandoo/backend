import re
import nltk
from nltk.corpus import stopwords

# Download stopwords sekali saat startup
def download_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

download_nltk_resources()

STOP_WORDS = set(stopwords.words("indonesian"))

# Kamu bisa menambahkan daftar kata gaul/slang tambahan di dictionary ini
NORM_DICT = {
    "bgt": "banget",
    "lemot": "lambat",
    "lemottt": "lambat",
    "gajelas": "tidak jelas",
    "ngentod": "kasar",
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "brg": "barang",
    "bgs": "bagus",
    "tp": "tapi",
    "tpi": "tapi",
    "jg": "juga",
    "jga": "juga",
    "tdk": "tidak",
    "aplikasinya": "aplikasi"
}


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


def normalize_text(text: str) -> str:
    """
    Langkah 2: Normalisasi Teks
    - Mengubah kata singkatan/slang menjadi kata baku
    """
    tokens = text.split()
    normalized_tokens = [NORM_DICT.get(word, word) for word in tokens]
    return " ".join(normalized_tokens)


def remove_stopwords(text: str) -> str:
    """
    Langkah 3: Stopword Removal (Bahasa Indonesia)
    """
    tokens = text.split()
    filtered_tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(filtered_tokens)


def stem_and_tokenize(text: str) -> tuple[str, list[str]]:
    """
    Langkah 4 & 5: Stemming & Tokenisasi
    - Mengembalikan string hasil stemming dan list token kata
    """
    tokens = text.split()
    stemmed_text = " ".join(tokens)  # Dapat diintegrasikan dengan Sastrawi Stemmer jika ada
    return stemmed_text, tokens


def get_full_preprocessing_steps(text: str) -> dict:
    """
    Mengalirkan teks melalui 5 alur preprocessing lengkap untuk UI & API response
    """
    original = text
    cleaned = clean_text(original)
    normalized = normalize_text(cleaned)
    no_stopword = remove_stopwords(normalized)
    stemmed, tokens = stem_and_tokenize(no_stopword)

    return {
        "original": original,
        "cleaning": cleaned,
        "normalized": normalized,
        "stopwords": no_stopword,
        "stemmed": stemmed,
        "tokens": tokens,
    }


# ─── Backward Compatibility ───────────────────────────────────────────────────

def preprocess_for_svm(text: str) -> tuple[str, str]:
    """
    Pipeline lama untuk SVM:
    Input → cleaning → stopword removal
    Return: (after_cleaning, after_stopword)
    """
    cleaned = clean_text(text)
    no_stopword = remove_stopwords(cleaned)
    return cleaned, no_stopword


def preprocess_for_indobert(text: str) -> str:
    """
    Pipeline untuk IndoBERT:
    Input → cleaning & normalisasi (TANPA stopword removal)
    IndoBERT butuh konteks kalimat yang lebih natural.
    """
    cleaned = clean_text(text)
    return normalize_text(cleaned)