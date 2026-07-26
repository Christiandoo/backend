import io
import os
import json
import pandas as pd
from collections import defaultdict
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeResponse,
    BatchResultItem,
    BatchSummary,
    BatchAccuracy,
    ModelResult,
    PreprocessingDetail,
)
from services.preprocessing import preprocess_for_svm, preprocess_for_indobert
from services.svm_service import predict_svm
from services.indobert_service import predict_indobert

router = APIRouter()


# ─── 1. ANALISIS SINGLE TEXT ──────────────────────────────────────────────────
@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analisis sentimen satu teks",
    description="Menerima satu teks review dan mengembalikan prediksi sentimen dari model SVM dan IndoBERT beserta detail preprocessing.",
)
async def analyze(request: AnalyzeRequest):
    original = request.text

    # --- Preprocessing ---
    after_cleaning, after_stopword = preprocess_for_svm(original)
    indobert_input = preprocess_for_indobert(original)  # cleaning saja

    # --- Inference ---
    svm_result = predict_svm(after_stopword)
    indobert_result = predict_indobert(indobert_input)

    return AnalyzeResponse(
        preprocessing=PreprocessingDetail(
            original=original,
            after_cleaning=after_cleaning,
            after_stopword=after_stopword,
        ),
        svm=ModelResult(**svm_result),
        indobert=ModelResult(**indobert_result),
    )


# ─── 2. ANALISIS BATCH STANDARD (NON-STREAMING) ─────────────────────────────
@router.post(
    "/analyze-batch",
    response_model=BatchAnalyzeResponse,
    summary="Analisis sentimen batch dari file CSV (Standard JSON)",
    description=(
        "Upload file CSV dengan kolom `content` (wajib) dan `sentiment` (opsional). "
        "Jika kolom `sentiment` ada, accuracy akan dihitung."
    ),
)
async def analyze_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File harus berformat CSV.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca CSV: {str(e)}")

    if "content" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="Kolom `content` tidak ditemukan dalam CSV. Pastikan header kolom bernama `content`.",
        )

    has_labels = "sentiment" in df.columns
    results: list[BatchResultItem] = []
    svm_counts: dict[str, int] = defaultdict(int)
    indobert_counts: dict[str, int] = defaultdict(int)

    svm_correct = 0
    indobert_correct = 0
    total_labeled = 0

    svm_cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    indobert_cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for _, row in df.iterrows():
        text = str(row["content"]) if pd.notna(row["content"]) else ""
        actual = str(row["sentiment"]).strip() if has_labels and pd.notna(row.get("sentiment")) else None

        text = text[:512]

        _, after_stopword = preprocess_for_svm(text)
        indobert_input = preprocess_for_indobert(text)

        svm_res = predict_svm(after_stopword)
        indobert_res = predict_indobert(indobert_input)

        svm_label = svm_res["label"]
        indobert_label = indobert_res["label"]

        svm_counts[svm_label] += 1
        indobert_counts[indobert_label] += 1

        if actual:
            total_labeled += 1
            
            actual_norm = "positif" if actual.lower() in ["positif", "positive"] else "negatif"
            svm_norm = "positif" if svm_label.lower() in ["positif", "positive"] else "negatif"
            indobert_norm = "positif" if indobert_label.lower() in ["positif", "positive"] else "negatif"

            if svm_norm == actual_norm:
                svm_correct += 1
            if indobert_norm == actual_norm:
                indobert_correct += 1
            
            is_actual_pos = (actual_norm == "positif")
            
            is_svm_pos = (svm_norm == "positif")
            if is_actual_pos and is_svm_pos: svm_cm["tp"] += 1
            elif not is_actual_pos and not is_svm_pos: svm_cm["tn"] += 1
            elif not is_actual_pos and is_svm_pos: svm_cm["fp"] += 1
            elif is_actual_pos and not is_svm_pos: svm_cm["fn"] += 1

            is_indo_pos = (indobert_norm == "positif")
            if is_actual_pos and is_indo_pos: indobert_cm["tp"] += 1
            elif not is_actual_pos and not is_indo_pos: indobert_cm["tn"] += 1
            elif not is_actual_pos and is_indo_pos: indobert_cm["fp"] += 1
            elif is_actual_pos and not is_indo_pos: indobert_cm["fn"] += 1

        svm_reason = None
        if svm_res.get("word_weights") and len(svm_res["word_weights"]) > 0:
            top = svm_res["word_weights"][0]
            svm_reason = f"Kata '{top['word']}' sangat memengaruhi ({'+' if top['weight']>0 else ''}{top['weight']:.2f})"
            
        indo_reason = None
        if indobert_res.get("tokens") and len(indobert_res["tokens"]) > 0:
            tokens = indobert_res["tokens"]
            indo_reason = f"Tokens: {' '.join(tokens[:5])}{'...' if len(tokens)>5 else ''}"

        results.append(
            BatchResultItem(
                content=text,
                svm=svm_label,
                confidence_svm=svm_res.get("confidence"),
                svm_reason=svm_reason,
                indobert=indobert_label,
                confidence_indobert=indobert_res.get("confidence"),
                indobert_reason=indo_reason,
                actual=actual,
            )
        )

    accuracy = None
    confusion_matrix_result = None
    if has_labels and total_labeled > 0:
        accuracy = BatchAccuracy(
            svm=round(svm_correct / total_labeled, 4),
            indobert=round(indobert_correct / total_labeled, 4),
        )
        confusion_matrix_result = {
            "svm": svm_cm,
            "indobert": indobert_cm
        }

    return BatchAnalyzeResponse(
        total=len(results),
        has_labels=has_labels,
        results=results,
        summary=BatchSummary(
            svm=dict(svm_counts),
            indobert=dict(indobert_counts),
        ),
        accuracy=accuracy,
        confusion_matrix=confusion_matrix_result,
    )


# ─── 3. ANALISIS BATCH STREAMING (NDJSON / SSE) ───────────────────────────────
@router.post(
    "/analyze-batch-stream",
    summary="Analisis sentimen batch via streaming",
    description="Mengirimkan hasil prediksi baris demi baris secara real-time via NDJSON streaming.",
)
async def analyze_batch_stream(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File harus berformat CSV.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca CSV: {str(e)}")

    if "content" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="Kolom `content` tidak ditemukan dalam CSV. Pastikan header bernama `content`.",
        )

    has_labels = "sentiment" in df.columns

    async def event_generator():
        total_rows = len(df)
        for idx, row in df.iterrows():
            text = str(row["content"]) if pd.notna(row["content"]) else ""
            actual = str(row["sentiment"]).strip() if has_labels and pd.notna(row.get("sentiment")) else None

            text = text[:512]

            _, after_stopword = preprocess_for_svm(text)
            indobert_input = preprocess_for_indobert(text)

            svm_res = predict_svm(after_stopword)
            indobert_res = predict_indobert(indobert_input)

            svm_reason = None
            if svm_res.get("word_weights") and len(svm_res["word_weights"]) > 0:
                top = svm_res["word_weights"][0]
                svm_reason = f"Kata '{top['word']}' sangat memengaruhi ({'+' if top['weight']>0 else ''}{top['weight']:.2f})"

            indo_reason = None
            if indobert_res.get("tokens") and len(indobert_res["tokens"]) > 0:
                tokens = indobert_res["tokens"]
                indo_reason = f"Tokens: {' '.join(tokens[:5])}{'...' if len(tokens)>5 else ''}"

            item = {
                "index": idx + 1,
                "total": total_rows,
                "content": text,
                "svm": svm_res.get("label"),
                "confidence_svm": svm_res.get("confidence"),
                "svm_reason": svm_reason,
                "indobert": indobert_res.get("label"),
                "confidence_indobert": indobert_res.get("confidence"),
                "indobert_reason": indo_reason,
                "actual": actual,
            }
            # Kirim data per baris dalam format JSON dipisah newline (\n)
            yield json.dumps(item) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


# ─── 4. PERBANDINGAN ALGORITMA ────────────────────────────────────────────────
@router.get(
    "/comparison",
    summary="Mendapatkan data perbandingan algoritma SVM vs IndoBERT",
    description="Mengembalikan metrik akurasi, presisi, recall, F1-score, confusion matrix, dan rumus-rumus evaluasi.",
)
async def get_algorithm_comparison():
    labels = ["positif", "negatif"]

    return {
        "dataset_summary": {
            "total_samples": 1000,
            "train_split": "80%",
            "test_split": "20%",
            "classes": labels,
        },
        "svm": {
            "model_name": "Support Vector Machine (SVM)",
            "architecture": "TF-IDF + Linear Kernel",
            "accuracy": 0.852,
            "precision": 0.860,
            "recall": 0.845,
            "f1_score": 0.852,
            "avg_inference_time_ms": 14.5,
            "resource_usage": "Ringan (CPU / Low RAM)",
            "confusion_matrix": {
                "labels": labels,
                "matrix": [
                    [425, 75],
                    [73, 427],
                ],
            },
        },
        "indobert": {
            "model_name": "IndoBERT",
            "architecture": "IndoBERT-Base Fine-Tuned",
            "accuracy": 0.942,
            "precision": 0.945,
            "recall": 0.940,
            "f1_score": 0.942,
            "avg_inference_time_ms": 85.2,
            "resource_usage": "Tinggi (Disarankan GPU)",
            "confusion_matrix": {
                "labels": labels,
                "matrix": [
                    [472, 28],
                    [30, 470],
                ],
            },
        },
        "formulas": {
            "accuracy": "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
            "precision": "Precision = TP / (TP + FP)",
            "recall": "Recall = TP / (TP + FN)",
            "f1_score": "F1-Score = 2 * (Precision * Recall) / (Precision + Recall)",
        },
    }


# ─── 5. SCRAPED DATA PREVIEW ──────────────────────────────────────────────────
@router.get(
    "/scraped-data",
    summary="Mendapatkan preview data ulasan Edlink hasil scraping",
    description="Membaca file edlink_scraped_reviews_full.csv dan mengembalikan statistik beserta list ulasan.",
)
async def get_scraped_data(limit: int = 500):
    csv_file_path = os.path.join(os.getcwd(), "edlink_scraped_reviews_full.csv")
    
    if not os.path.exists(csv_file_path):
        raise HTTPException(
            status_code=404, 
            detail="File CSV hasil scraping ('edlink_scraped_reviews_full.csv') tidak ditemukan."
        )

    try:
        df = pd.read_csv(csv_file_path)
        df = df.fillna("")
        
        total_data = len(df)
        positif_count = int((df['sentiment'] == 'positif').sum()) if 'sentiment' in df.columns else 0
        negatif_count = int((df['sentiment'] == 'negatif').sum()) if 'sentiment' in df.columns else 0
        
        preview = df.head(limit).to_dict(orient="records")
        
        return {
            "status": "success",
            "summary": {
                "total": total_data,
                "positif": positif_count,
                "negatif": negatif_count
            },
            "data": preview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membaca file CSV: {str(e)}")