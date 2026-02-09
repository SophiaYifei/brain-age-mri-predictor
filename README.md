# brain-age-mri-predictor
Predict brain age from MRI scans using a mix of deep learning and classical ML models. This repository contains data processing helpers, training and inference scripts, pretrained weights, and a small FastAPI inference service.

**Features**
- **Multi‑modality:** Supports T1, T2, PD, and MRA modalities used in MRI brain imaging.
- **Model types:** Naive baselines, classical ML models, single‑modality deep models, and a late‑fusion deep model.
- **Inference Application:** Web application to upload one image or all 4 image types and receive a predicted age on a T2-trained DL model and the late-fusion DL model.

**Quick Links**
- **Requirements:** [requirements.txt](requirements.txt)
- **Setup / helpers:** [setup.py](setup.py)
- **Application:** https://brain-age-mri-predictor-production.up.railway.app/

**Repository layout (high level)**
- `data/`: raw images(`data/raw/`) and label CSVs(`data/labels/`).
- `scripts/`: training, dataset building, inference helpers (see `scripts/model.py` and  `scripts/fusion_model.py`).
- `models/`: trained `.pth` weights used for inference and experiments, can run `scripts/download_weights.py` to save inference models in `models/`.
- `main.py`: script to run web application.
- `setup.py`: script to setup data and train models.
- `static/`: files for web application.
- `notebooks/`: exploratory analysis and notebooks used during the project.

**Setup Repository**
Install system dependencies and Python packages from the provided file.

```bash
git clone https://github.com/SophiaYifei/brain-age-mri-predictor.git
cd brain-age-mri-predictor

# Create and activate a virtual environment (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# Install Python deps
pip install -r requirements.txt
```

Some project setup (data download or model training/weight fetching) is available via `setup.py` if needed:

```bash
python setup.py
```

**Data**
- The dataset used is the IXI dataset (https://brain-development.org/ixi-dataset/). 
- CSV label files and train/val/test splits are in `data/labels/`.
- Raw PNG image folders are available under `data/raw/`. They are loaded from a google cloud storage in `setup.py`.

**Models**
- Pretrained weights will be stored under `models/` (files ending in `.pth`). 
- Run `scripts/download_weights.py` to download pre-trained models to be saved in `models/`.
- Run `setup.py` to retrain models to be saved in `models/`.

**Web Application**
Deployed through Railway on https://brain-age-mri-predictor-production.up.railway.app/

There are two inference models that users can try. 

There is the Quick Scan T2 Brain Age Predictor, where the user must upload a T2 image and click "Analyze T2 Scan" to receive an estimated brain age. If the user does not have an image, they can utilize the given sample patient data on the website.

Also, there is the Full Analysis Fusion Brain Age Prediction, where the user must upload all image modalities (T1, T2, PD, MRA) and click "Run Fusion Analysis" to receive an estimated brain age. If the user does not have images, they can utilize the given sample patient data for all 4 modalities on the website.



**Authors**
- Diya Mirji
- Yifei Guo
- Omkar Sreekanth

**License**
- This repository includes a `LICENSE` file at the project root.
