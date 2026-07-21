import joblib
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "svm"

# ─── Load model & vectorizer saat startup (singleton) ────────────────────────
_svm_model = joblib.load(MODEL_DIR / "svm_model.pkl")
_tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
_label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")


def _sigmoid(x: float) -> float:
    """Konversi decision function score ke pseudo-probability [0,1] via sigmoid."""
    return float(1.0 / (1.0 + np.exp(-x)))


def predict_svm(text_after_stopword: str) -> dict:
    """
    Menerima teks setelah preprocessing (cleaning + stopword removal).
    LinearSVC tidak punya predict_proba, jadi confidence dihitung
    dari decision_function → sigmoid.

    Return: { label: str, confidence: float }
    """
    vec = _tfidf.transform([text_after_stopword])

    # Predict label
    pred_encoded = _svm_model.predict(vec)[0]
    raw_label = _label_encoder.inverse_transform([pred_encoded])[0]
    
    # Map to Indonesian as specified in PLAN.md
    label = "positif" if raw_label == "positive" else "negatif"

    # Confidence via decision_function → sigmoid
    decision = _svm_model.decision_function(vec)[0]

    # Binary case: scalar, multi-class: array
    if np.ndim(decision) == 0:
        # Nilai positif → "positive", negatif → "negative"
        # Confidence = sigmoid dari abs(decision) atau dari arah yang benar
        confidence = _sigmoid(float(decision)) if raw_label == "positive" else _sigmoid(-float(decision))
    else:
        # Multi-class: ambil max score
        confidence = _sigmoid(float(np.max(decision)))

    # Extract feature weights
    word_weights = []
    if hasattr(_svm_model, "coef_") and hasattr(_tfidf, "get_feature_names_out"):
        feature_names = _tfidf.get_feature_names_out()
        # Binary classification usually has shape (1, n_features)
        coef = _svm_model.coef_[0] if np.ndim(_svm_model.coef_) > 1 else _svm_model.coef_
        nonzero_indices = vec.nonzero()[1]
        for idx in nonzero_indices:
            tfidf_val = vec[0, idx]
            weight = coef[idx] * tfidf_val
            word_weights.append({
                "word": feature_names[idx],
                "weight": float(weight)
            })
        
        # Sort by absolute weight magnitude to find most impactful words
        word_weights.sort(key=lambda x: abs(x["weight"]), reverse=True)
        word_weights = word_weights[:10]

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "word_weights": word_weights,
    }
