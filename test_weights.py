import joblib
import numpy as np
from pathlib import Path

MODEL_DIR = Path("model/svm")
_svm_model = joblib.load(MODEL_DIR / "svm_model.pkl")
_tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

print("SVM Model type:", type(_svm_model))
if hasattr(_svm_model, 'coef_'):
    print("Has coef_:", _svm_model.coef_.shape)
if hasattr(_tfidf, 'get_feature_names_out'):
    print("Vocab size:", len(_tfidf.get_feature_names_out()))
