import torch
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification

# Fallback ke repository IndoBERT di Hugging Face jika model lokal tidak lengkap
HF_MODEL_NAME = "indobenchmark/indobert-base-p1"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "indobert"

# Periksa apakah file config.json benar-benar ada di dalam folder lokal
if (LOCAL_MODEL_DIR / "config.json").exists():
    MODEL_PATH = str(LOCAL_MODEL_DIR)
else:
    MODEL_PATH = HF_MODEL_NAME

# ─── Label mapping ────────────────────────────────────────────────────────────
_LABEL_MAP = {
    0: "negatif",
    1: "positif",
}

# ─── Load model & tokenizer sekali saat startup (singleton) ───────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
_model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
_model.to(_device)
_model.eval()


def predict_indobert(text_after_cleaning: str) -> dict:
    """
    Menerima teks setelah cleaning (TANPA stopword removal).
    Return: { label: str, confidence: float }
    """
    inputs = _tokenizer(
        text_after_cleaning,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits  # shape: (1, num_labels)

    proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])

    # Selalu gunakan _LABEL_MAP manual kita untuk konsistensi
    label = _LABEL_MAP.get(pred_idx, str(pred_idx))

    # Ekstrak token untuk ditampilkan di UI
    tokens = _tokenizer.tokenize(text_after_cleaning)

    return {
        "label": str(label),
        "confidence": round(confidence, 4),
        "tokens": tokens,
    }