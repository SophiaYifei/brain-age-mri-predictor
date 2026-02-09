#AI used: Gemini 3 https://gemini.google.com/share/514144793393

import os
from huggingface_hub import hf_hub_download

# Your Hugging Face Repo ID
HF_REPO_ID = "dvm14/brain-age-predictor"

# The directory where main.py expects to find models
# (Relative to the root of your project)
MODELS_DIR = "models"

# Exact filenames you uploaded to Hugging Face
FILES_TO_DOWNLOAD = [
    "dl_model_T2.pth", 
    "final_late_fusion_model.pth"
]

def download_models():
    """Check for and download model weights from Hugging Face if not already present."""
    print(f"Checking for model weights from {HF_REPO_ID}...")
    
    # Ensure the models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for filename in FILES_TO_DOWNLOAD:
        local_path = os.path.join(MODELS_DIR, filename)
        
        # Check if file already exists to avoid re-downloading
        if os.path.exists(local_path):
            print(f"Found {filename} locally. Skipping download.")
        else:
            print(f"Downloading {filename}...")
            try:
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    local_dir=MODELS_DIR
                )
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                # Exit with error so the deployment fails immediately if weights are missing
                exit(1)

    print("All inference models are ready!")

#if __name__ == "__main__":
#    download_models()