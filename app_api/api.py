from fastapi import FastAPI, UploadFile, File
import tempfile
import os

# Import the inference-only helper you added
from scripts.model import predict_age_from_file

# Create the FastAPI app
app = FastAPI(title="Brain Age MRI Predictor API")

# Simple endpoint to confirm the API is running
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint that accepts an MRI image and returns predicted brain age
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an uploaded MRI image file,
    saves it temporarily,
    runs model inference,
    and returns the predicted age.
    """

    # Keep original file extension (e.g. .png, .jpg)
    suffix = os.path.splitext(file.filename)[1]

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Run inference using the existing model code
    prediction = predict_age_from_file(
        image_path=tmp_path,
        model_weights_path="models/dl_model_T1.pth",  # adjust modality later if needed
    )

    # Return result to frontend
    return {
        "predicted_age": prediction
    }
