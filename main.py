from fastapi import FastAPI
from contextlib import asynccontextmanager
import os

from api.routes.analyze import router as analyze_router
from ml.model import train_and_save, load_model, MODEL_PATH
from core.state import ml_state   # ✅ IMPORTANT (state.py, not ml_state.py)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train only first time
    if not os.path.exists(MODEL_PATH):
        train_and_save()

    # Load model into shared state
    ml_state["pipeline"] = load_model()
    print("✅ ML model loaded")

    yield

    ml_state.clear()
    print("✅ ML model cleared")


app = FastAPI(
    title="AI Code Analyzer",
    description="Analyze code snippets using AI + ML",
    version="1.0.0",
    lifespan=lifespan
)

# Include routes AFTER app creation
app.include_router(analyze_router, prefix="/api")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": "pipeline" in ml_state
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)