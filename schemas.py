from pydantic import BaseModel, Field
from typing import Optional


# ─── Request schemas ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Teks review yang akan dianalisis (maks 512 karakter)",
        examples=["Aplikasi ini sangat membantu belajar, fiturnya lengkap!"],
    )


# ─── Response schemas ──────────────────────────────────────────────────────────

class WordWeight(BaseModel):
    word: str
    weight: float


class ModelResult(BaseModel):
    label: str
    confidence: float
    word_weights: Optional[list[WordWeight]] = None
    tokens: Optional[list[str]] = None


class PreprocessingDetail(BaseModel):
    original: str
    after_cleaning: str
    after_stopword: str


class AnalyzeResponse(BaseModel):
    preprocessing: PreprocessingDetail
    svm: ModelResult
    indobert: ModelResult


# ─── Batch schemas ─────────────────────────────────────────────────────────────

class BatchResultItem(BaseModel):
    content: str
    svm: str
    confidence_svm: Optional[float] = None
    svm_reason: Optional[str] = None
    indobert: str
    confidence_indobert: Optional[float] = None
    indobert_reason: Optional[str] = None
    actual: Optional[str] = None


class BatchSummary(BaseModel):
    svm: dict[str, int]
    indobert: dict[str, int]


class BatchAccuracy(BaseModel):
    svm: Optional[float] = None
    indobert: Optional[float] = None


class ConfusionMatrix(BaseModel):
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0


class BatchAnalyzeResponse(BaseModel):
    total: int
    has_labels: bool
    results: list[BatchResultItem]
    summary: BatchSummary
    accuracy: Optional[BatchAccuracy] = None
    confusion_matrix: Optional[dict[str, ConfusionMatrix]] = None

    
