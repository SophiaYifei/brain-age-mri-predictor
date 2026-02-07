from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File
import tempfile
import os
from pathlib import Path

# ------------------------------------------------------------
# PURPOSE OF THIS FILE
# ------------------------------------------------------------
# This file exposes our brain-age model as a web API using FastAPI.
# The API is INFERENCE ONLY:
#   - takes an uploaded MRI image (PNG/JPG)
#   - loads trained model weights (.pth)
#   - runs one forward pass to predict brain age
#   - returns the result as JSON
# No training happens here.
# ------------------------------------------------------------

# Import the single-image inference function from our ML code.
# This function should:
#   1) build the model architecture (ResNet50 regressor)
#   2) load the learned weights from a .pth file
#   3) preprocess the uploaded image
#   4) return one number: predicted brain age
from scripts.model import predict_age_from_file

# Create the FastAPI app (this is the backend service object)
app = FastAPI(title="Brain Age MRI Predictor API")

# Repo root folder (so we can build paths reliably even if run from different directories)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Where we expect the trained weights to be stored locally.
# When teammates finish training, place the .pth file here (or update this path).
WEIGHTS_PATH = REPO_ROOT / "models" / "dl_model_T1.pth"

# If someone visits "/", show a helpful message instead of "Not Found"
@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "Brain Age API is running. Go to /docs to test."}

# Health check for uptime monitoring
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

# Status endpoint: shows whether weights exist and where the API is looking for them
@app.get("/status")
def status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "weights_found": WEIGHTS_PATH.exists(),
        "weights_path": str(WEIGHTS_PATH),
    }

# Version endpoint: useful for confirming what build is deployed
@app.get("/version")
def version() -> Dict[str, Any]:
    return {
        "service": "brain-age-api",
        "api_version": "0.1.0",
    }

# Prediction endpoint: upload one image and get a predicted age
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    # If weights are missing, return a readable message instead of crashing
    if not WEIGHTS_PATH.exists():
        return {
            "error": "Model weights not available yet.",
            "why": "The API needs the trained .pth weights file to run inference.",
            "next_step": "Place the trained weights at the weights_path below (or update WEIGHTS_PATH).",
            "weights_path": str(WEIGHTS_PATH),
        }

    # Keep the file extension (helps loaders correctly interpret the file)
    suffix = os.path.splitext(file.filename)[1]

    # Save uploaded file to a temporary path on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Run inference using the uploaded image + the trained weights file
        prediction = predict_age_from_file(
            image_path=tmp_path,
            model_weights_path=str(WEIGHTS_PATH),
        )
        return {"predicted_age": float(prediction)}
    finally:
        # Always attempt to delete temp file (even if inference fails)
        try:
            os.remove(tmp_path)
        except Exception:
            pass