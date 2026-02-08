# brain-age-mri-predictor
A computer vision system that predicts brain age from MRI scans using Deep Learning, Classical ML, and Naive baselines. Course project for AIPI540 Deep Learning.

## Setup
Clone the repo, create a virtual environment, and install dependencies.

```bash
git clone https://github.com/SophiaYifei/brain-age-mri-predictor.git
cd brain-age-mri-predictor

# Create and activate a virtual environment (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# To install data from cloud storage, and to train and save all model types
    # Naive Model (4 models trained for each of the 4 modality datasets)
    # Classical Model (4 models trained for each of the 4 modality datasets)
    # Deep Learning Model (4 models trained for each of the 4 modality datasets)
    # Deep Learning Late Fusion Model (1 model trained)
python setup.py

# To run application
#[PLACEHOLDER]
```

## Data
Data has been downloaded from the IXI dataset from https://brain-development.org/ixi-dataset/.

- We have stored the images from the IXI data in a google cloud storage bucket.
- To download the data from the google cloud storage, run setup.py

## Models
[PLACEHOLDER]