import model

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import importlib.util
from pathlib import Path
import make_dataset
import build_features

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image


# ======================== Grad-CAM Visualization ========================

class RegressionTarget:
    def __init__(self):
        pass
    def __call__(self, model_output):
        # model_output is the predicted age. 
        # We return it directly so the gradient is calculated for the value itself.
        return model_output


def get_gradcam_viz(model, input_tensor, original_img):
    # For ResNet50, the last convolutional layer is the end of layer4
    target_layers = [model.layer4[-1]]

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # For regression, we use RegressionTarget. 
    # This tells Grad-CAM to compute gradients w.r.t. the scalar output.
    targets = [RegressionTarget()]

    # Generate grayscale CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Overlay the heatmap on the original image (normalized 0-1)
    visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)
    return visualization

def run_gradcam_comparison(patient_image_paths, transform, true_age):
    """
    patient_image_paths: Dictionary { 'T1': 'path/to/img', 'T2': ... }
    transform: The preprocessing pipeline used during validation
    true_age: The actual age of the patient (for metrics)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    modalities = ['T1'] #, 'T2', 'PD', 'MRA']
    
    for i, modality in enumerate(modalities):
        # 1. Reconstruct Architecture
        model = resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1) # Matches your training setup
        
        # 2. Load Weights
        model_path = f'../models/dl_model_{modality}_skull.pth'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # 3. Load and Preprocess Image
        img_path = patient_image_paths[modality]
        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0) # Add batch dimension

        # 3. Get Prediction
        with torch.no_grad():
            pred_age = model(input_tensor).item()
        
        # 4. Calculate Single-Image Metrics
        mae = abs(pred_age - true_age)
        rmse = np.sqrt((pred_age - true_age)**2) # For 1 sample, RMSE == MAE, but included for logic
        
        # Create a float version of the original image for visualization (0 to 1)
        # We resize it to match the input_tensor size (224x224)
        rgb_img = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0

        # 5. Generate Visualization
        viz = get_gradcam_viz(model, input_tensor, rgb_img)
        
        # 6. Plotting
        axes[i].imshow(viz)
        axes[i].set_title(f"{modality}\nTrue: {true_age:.1f} | Pred: {pred_age:.1f}")
        
        # Add a text box for metrics
        stats_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}"
        axes[i].text(10, 210, stats_text, color='white', weight='bold', 
                     bbox=dict(facecolor='black', alpha=0.5))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_gradcam_comparison({
        'T1': '../data/raw/IXI_T1_png/IXI002-Guys-0828-T1.png',
        'T2': '../data/raw/IXI_T2_png/IXI002-Guys-0828-T2.png',
        'PD': '../data/raw/IXI_PD_png/IXI002-Guys-0828-PD.png',
        'MRA': '../data/raw/IXI_MRA_png/IXI002-Guys-0828-MRA.png'}, 
        model.build_transforms()[2], true_age=35.80013689253936)
    
    run_gradcam_comparison({
        'T1': '../data/raw/IXI_T1_png/IXI013-HH-1212-T1.png',
        'T2': '../data/raw/IXI_T2_png/IXI013-HH-1212-T2.png',
        'PD': '../data/raw/IXI_PD_png/IXI013-HH-1212-PD.png',
        'MRA': '../data/raw/IXI_MRA_png/IXI013-HH-1212-MRA.png'}, 
        model.build_transforms()[2], true_age=46.71047227926078)
    
    run_gradcam_comparison({
        'T1': '../data/raw/IXI_T1_png/IXI019-Guys-0702-T1.png',
        'T2': '../data/raw/IXI_T2_png/IXI019-Guys-0702-T2.png',
        'PD': '../data/raw/IXI_PD_png/IXI019-Guys-0702-PD.png',
        'MRA': '../data/raw/IXI_MRA_png/IXI019-Guys-0702-MRA.png'}, 
        model.build_transforms()[2], true_age=58.65845311430527)