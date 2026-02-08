from app_api.api import app

# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import os
from pathlib import Path
from typing import List, Optional

# Import your model logic
# Import T2 logic from model.py
from scripts.model import predict_t2_model

# Import Fusion logic from fusion_model.py
from scripts.fusion_model import predict_fusion_model

app = FastAPI()

# Define paths
REPO_ROOT = Path(__file__).resolve().parent
STATIC_PATH = REPO_ROOT / "static"
SAMPLES_PATH = STATIC_PATH / "samples"
MODELS_PATH = REPO_ROOT / "models"

# 1. SERVE THE FRONTEND
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

@app.get("/")
async def read_index():
    return FileResponse(str(STATIC_PATH / "index.html"))


SAMPLE_META = {
    "t2": {
        "age": 46.71, 
        "path": "/static/samples/sample_T2.png"
    },
    "fusion": {
        "age": 46.71,
        # Dictionary of paths for the grid view
        "paths": {
            "T1": "/static/samples/sample_T1.png",
            "T2": "/static/samples/sample_T2.png",
            "PD": "/static/samples/sample_PD.png",
            "MRA": "/static/samples/sample_MRA.png"
        }
    }
}



# 2. PREDICT ENDPOINT: T2 ONLY
@app.post("/predict/t2")
async def predict_t2(
    file: Optional[UploadFile] = File(None), 
    use_sample: bool = Form(False)
):
    temp_path = None
    true_age = None
    image_url = None

    try:
        if use_sample:
            temp_path = SAMPLES_PATH / "sample_T2.png" 
            if not temp_path.exists():
                return {"error": "Sample files missing.", "status": "failed"}
            inference_path = str(temp_path)
            
            # Set metadata for response
            true_age = SAMPLE_META["t2"]["age"]
            image_url = SAMPLE_META["t2"]["path"]
        else:
            if not file:
                return {"error": "No file uploaded.", "status": "failed"}
            
            suffix = Path(file.filename).suffix
            temp_path = Path(f"temp_t2{suffix}")
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            inference_path = str(temp_path)
            # True age is unknown for user uploads
            true_age = None 

        # INFERENCE
        predicted_age = predict_t2_model(inference_path)
        
        # Cleanup
        if not use_sample and temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "status": "success",
            "age": float(predicted_age),
            "true_age": true_age,
            "image_url": image_url, # Only populated if sample
            "is_sample": use_sample
        }

    except Exception as e:
        if temp_path and not use_sample and os.path.exists(temp_path):
             os.remove(temp_path)
        return {"error": str(e), "status": "failed"}

# 3. PREDICT ENDPOINT: FUSION (4 Files)
@app.post("/predict/fusion")
async def predict_fusion(
    t1: Optional[UploadFile] = File(None),
    t2: Optional[UploadFile] = File(None),
    pd: Optional[UploadFile] = File(None),
    mra: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False)
):
    temp_files = {}
    true_age = None
    image_urls = None # Will hold the dictionary of paths

    try:
        paths = {}
        
        if use_sample:
            # Use paths from metadata
            true_age = SAMPLE_META["fusion"]["age"]
            image_urls = SAMPLE_META["fusion"]["paths"]
            
            # Map keys to local file paths for inference
            # Note: We strip the leading '/' for local file access if needed, 
            # but usually, your inference script needs absolute or relative file paths.
            # Assuming your static folder is in root:
            paths['T1'] = "static/samples/sample_T1.png"
            paths['T2'] = "static/samples/sample_T2.png"
            paths['PD'] = "static/samples/sample_PD.png"
            paths['MRA'] = "static/samples/sample_MRA.png"
            
            # verify existence
            for p in paths.values():
                if not os.path.exists(p):
                     return {"error": f"Sample file missing: {p}", "status": "failed"}

        else:
            if not all([t1, t2, pd, mra]):
                return {"error": "All 4 modalities are required.", "status": "failed"}

            def save_upload(upl_file, key):
                path = Path(f"temp_fusion_{key}{Path(upl_file.filename).suffix}")
                with open(path, "wb") as b:
                    shutil.copyfileobj(upl_file.file, b)
                return str(path)

            paths['T1'] = save_upload(t1, "t1")
            paths['T2'] = save_upload(t2, "t2")
            paths['PD'] = save_upload(pd, "pd")
            paths['MRA'] = save_upload(mra, "mra")
            temp_files = paths
            
            # For uploads, image_urls will be null, frontend handles preview locally
            image_urls = None 

        # INFERENCE
        predicted_age = predict_fusion_model(paths)

        # Cleanup
        if not use_sample:
            for p in temp_files.values():
                if os.path.exists(p): os.remove(p)

        return {
            "status": "success",
            "age": float(predicted_age),
            "true_age": true_age,
            "image_urls": image_urls, # Sends back dictionary of 4 paths (if sample)
            "is_sample": use_sample
        }

    except Exception as e:
        for p in temp_files.values():
            if os.path.exists(p): os.remove(p)
        return {"error": str(e), "status": "failed"}