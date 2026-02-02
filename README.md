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

# To install data from cloud storage
python setup.py
```

## Data
Download the IXI dataset from https://brain-development.org/ixi-dataset/.

- Choose **T1 images (all images)**.
- Extract the archive.
- Place the extracted folder under `data/raw/IXI_T1/`.

After that, you can run the notebooks in `notebooks/`.
