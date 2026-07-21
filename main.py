import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.analyze import router as analyze_router


# ─── Lifespan: model loading di startup ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load semua model saat startup.
    Service modules (svm_service, indobert_service) sudah melakukan
    import-time loading, sehingga cukup import untuk memicu loading.
    """
    print("Loading models...")
    t0 = time.time()

    # Import memicu singleton loading
    import services.svm_service  # noqa: F401
    import services.indobert_service  # noqa: F401

    print(f"Models loaded in {time.time() - t0:.2f}s")
    yield
    print("Shutting down...")


# ─── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API — Edlink Reviews",
    description=(
        "API analisis sentimen ulasan aplikasi Edlink menggunakan dua model ML: "
        "**SVM** (TF-IDF + SVC) dan **IndoBERT** (BertForSequenceClassification). "
        "Dibuat untuk keperluan skripsi."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Ganti dengan domain spesifik saat production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────
app.include_router(analyze_router, tags=["Sentiment Analysis"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint — dibutuhkan Railway untuk monitoring."""
    return {"status": "ok"}


# ─── Entrypoint (python main.py) ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
