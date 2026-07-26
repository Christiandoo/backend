import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── Configuration ────────────────────────────────────────────────────────────
# Path repositori model kamu yang sudah valid di Hugging Face
HF_MODEL_NAME = "itsmedo/indoberet"

# ─── Label mapping ────────────────────────────────────────────────────────────
_LABEL_MAP = {
    0: "negatif",
    1: "positif",
}

# ─── Load model & tokenizer sekali saat startup (singleton) ───────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Menggunakan AutoClass agar kompatibel penuh dengan config di Hugging Face
_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)

_model.to(_device)
_model.eval()


def predict_indobert(text_after_cleaning: str) -> dict:
    """
    Menerima teks setelah cleaning.
    Return: { label: str, confidence: float, tokens: list }
    """
    # Validasi input kosong/hanya whitespace
    if not text_after_cleaning or not text_after_cleaning.strip():
        return {
            "label": "netral",
            "confidence": 0.0,
            "tokens": [],
        }

    # Process tokenization (cukup 1x pemanggilan)
    inputs = _tokenizer(
        text_after_cleaning,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    
    # Ambil daftar token langsung dari hasil tokenizer tanpa memproses ulang
    tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Pindahkan tensor ke device (CPU/GPU)
    inputs_on_device = {k: v.to(_device) for k, v in inputs.items()}

    # Inference tanpa menghitung gradient
    with torch.no_grad():
        outputs = _model(**inputs_on_device)
        logits = outputs.logits  # shape: (1, num_labels)

    # Hitung probabilitas menggunakan softmax
    proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])

    # Ambil label berdasarkan mapping
    label = _LABEL_MAP.get(pred_idx, str(pred_idx))

    return {
        "label": str(label),
        "confidence": round(confidence, 4),
        "tokens": tokens,
    }