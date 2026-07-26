import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── Optimasi CPU Threads untuk Railway ───────────────────────────────────────
# Mencegah CPU throttling/context-switching di environment server terbatas
torch.set_num_threads(2)

# ─── Configuration ────────────────────────────────────────────────────────────
HF_MODEL_NAME = "itsmedo/indoberet"

# ─── Label mapping ────────────────────────────────────────────────────────────
_LABEL_MAP = {
    0: "negatif",
    1: "positif",
}

# ─── Load model & tokenizer sekali saat startup (singleton) ───────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)

_model.to(_device)
_model.eval()


def predict_indobert(text_after_cleaning: str) -> dict:
    """
    Menerima teks setelah cleaning.
    Return: { label: str, confidence: float, tokens: list }
    """
    # 1. Validasi input kosong atau hanya spasi
    if not text_after_cleaning or not text_after_cleaning.strip():
        return {
            "label": "netral",
            "confidence": 0.0,
            "tokens": [],
        }

    # 2. Proses Tokenisasi (max_length diturunkan ke 128 agar jauh lebih cepat di CPU)
    inputs = _tokenizer(
        text_after_cleaning,
        return_tensors="pt",
        truncation=True,
        max_length=128,  # <--- OPTIMASI UTAMA: Turunkan dari 512 ke 128
    )

    # 3. Ekstrak token & filter token khusus ([CLS], [SEP], [PAD])
    raw_tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    clean_tokens = [
        token for token in raw_tokens 
        if token not in (_tokenizer.cls_token, _tokenizer.sep_token, _tokenizer.pad_token)
    ]

    # 4. Pindahkan input tensor ke device (CPU/GPU)
    inputs_on_device = {k: v.to(_device) for k, v in inputs.items()}

    # 5. Inference (Prediksi)
    with torch.no_grad():
        outputs = _model(**inputs_on_device)
        logits = outputs.logits  # shape: (1, num_labels)

    # 6. Hitung probabilitas dengan Softmax
    proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])

    # 7. Pemetaan ke label teks
    label = _LABEL_MAP.get(pred_idx, str(pred_idx))

    return {
        "label": str(label),
        "confidence": round(confidence, 4),
        "tokens": clean_tokens,
    }