import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# ---------------------------------------------------------------------------
# Feature order — must stay consistent across training and prediction
# ---------------------------------------------------------------------------

FEATURES = ["number_of_functions", "average_function_length", "max_nesting_depth"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "quality_model.joblib")


# ---------------------------------------------------------------------------
# 1. Dummy training data
# ---------------------------------------------------------------------------
# Each row is [number_of_functions, average_function_length, max_nesting_depth]
# Scores reflect reasonable heuristics:
#   - More functions = better decomposition → higher score
#   - Very long functions → lower score
#   - Deep nesting → lower score

def _build_training_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y) where:
        X — feature matrix  shape (n_samples, 3)
        y — quality scores  shape (n_samples,)  range 0–100
    """
    # [num_functions, avg_length, max_nesting]  →  quality_score
    samples = [
        # ── pristine code ───────────────────────────────────────────────
        ([10,  5.0, 1],  95),   # many small, shallow functions
        ([8,   6.0, 1],  92),
        ([6,   8.0, 2],  88),
        ([5,   7.0, 2],  85),
        ([7,   9.0, 2],  83),
        ([4,  10.0, 2],  80),
        # ── average code ────────────────────────────────────────────────
        ([3,  15.0, 3],  70),
        ([4,  18.0, 3],  65),
        ([2,  20.0, 3],  60),
        ([3,  22.0, 3],  58),
        ([2,  25.0, 3],  55),
        ([1,  20.0, 3],  50),
        # ── poor code ───────────────────────────────────────────────────
        ([1,  35.0, 4],  40),
        ([2,  40.0, 4],  35),
        ([1,  45.0, 5],  28),
        ([1,  50.0, 5],  22),
        ([1,  60.0, 6],  15),
        # ── worst cases ─────────────────────────────────────────────────
        ([1,  80.0, 7],  10),
        ([0,   0.0, 0],  50),   # empty file → neutral
        ([1, 100.0, 8],   5),
    ]

    X = np.array([s[0] for s in samples], dtype=float)
    y = np.array([s[1] for s in samples], dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# 2. Model training
# ---------------------------------------------------------------------------

def train() -> Pipeline:
    """
    Build and fit a GradientBoostingRegressor inside a scaling pipeline.

    Pipeline steps:
        MinMaxScaler  — normalises features to [0, 1] so the model
                        isn't thrown off by scale differences between
                        (num_functions ≈ 1–10) vs (avg_length ≈ 5–100).
        GradientBoostingRegressor — handles non-linear interactions
                        (e.g. many functions AND shallow nesting = good)
                        while remaining interpretable via feature importance.
    """
    X, y = _build_training_data()

    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=100,   # enough trees for a smooth score surface
            max_depth=3,        # shallow trees → less overfitting on small data
            learning_rate=0.1,
            random_state=42,
        )),
    ])

    pipeline.fit(X, y)
    return pipeline


def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    joblib.dump(pipeline, path)


def load_model(path: str = MODEL_PATH) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model at '{path}'. Call train_and_save() first."
        )
    return joblib.load(path)


def train_and_save(path: str = MODEL_PATH) -> Pipeline:
    """Convenience: train, persist, and return the pipeline."""
    pipeline = train()
    save_model(pipeline, path)
    return pipeline


# ---------------------------------------------------------------------------
# 3. Prediction
# ---------------------------------------------------------------------------

def predict_quality(features: dict, pipeline: Pipeline) -> dict:
    """
    Predict a code quality score from extracted features.

    Args:
        features: output of analyzer.features.extract_features()
                  {"number_of_functions": int,
                   "average_function_length": float,
                   "max_nesting_depth": int}
        pipeline: a trained sklearn Pipeline (from train() or load_model())

    Returns:
        {
            "quality_score": float,          # 0–100, two decimal places
            "grade":         str,            # A / B / C / D / F
            "feature_importance": {          # which features mattered most
                "number_of_functions":    float,
                "average_function_length": float,
                "max_nesting_depth":      float,
            }
        }
    """
    X = np.array([[features[f] for f in FEATURES]])
    raw_score = pipeline.predict(X)[0]
    score = float(np.clip(raw_score, 0, 100))

    return {
        "quality_score":      round(score, 2),
        "grade":              _score_to_grade(score),
        "feature_importance": _get_importance(pipeline),
    }


def _score_to_grade(score: float) -> str:
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"


def _get_importance(pipeline: Pipeline) -> dict[str, float]:
    """Extract normalised feature importances from the GBR model."""
    importances = pipeline.named_steps["model"].feature_importances_
    total = importances.sum() or 1.0      # guard against all-zero edge case
    return {
        feature: round(float(imp / total), 4)
        for feature, imp in zip(FEATURES, importances)
    }