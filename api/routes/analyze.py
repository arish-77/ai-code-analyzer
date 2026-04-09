from fastapi import APIRouter, HTTPException

# ✅ Import schemas (NEW)
from api.schemas.request import AnalyzeRequest
from api.schemas.response import AnalyzeResponse

# ✅ Core logic
from analyzer.parser import analyze
from analyzer.features import extract_features
from ml.model import predict_quality

# ✅ Shared ML state
from core.state import ml_state


router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_code(payload: AnalyzeRequest):
    """
    Pipeline:
    1. AST analysis
    2. Feature extraction
    3. ML scoring
    """

    # 🔹 Step 1: AST Analysis
    try:
        ast_result = analyze(payload.code)
    except SyntaxError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid Python syntax: {exc.msg} (line {exc.lineno})",
        )

    # 🔹 Step 2: Feature Extraction
    features = extract_features(payload.code)

    # 🔹 Step 3: ML Prediction
    prediction = predict_quality(features, ml_state["pipeline"])

    # 🔹 Final Response
    return AnalyzeResponse(
        issues=ast_result["issues"],
        score=prediction["quality_score"],
        grade=prediction["grade"],
        feature_importance=prediction["feature_importance"],
    )