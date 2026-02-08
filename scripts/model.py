import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNet
import pickle

import matplotlib.pyplot as plt
import importlib.util
from pathlib import Path

from scripts import make_dataset
from scripts import build_features

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

from google.cloud import storage

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Path Setup ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
PATH_T2 = os.path.join(MODELS_DIR, "dl_model_T2.pth")

# --- Architecture ---
def build_resnet50_regressor():
    """Helper to build a ResNet50 with 1 output node"""
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# --- Helper ---
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

# --- Inference Function ---
def predict_t2_model(image_path, weights_path=None):
    if weights_path is None:
        weights_path = PATH_T2

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}. Run scripts/download_weights.py first.")

    device = torch.device("cpu")
    # --- DEBUGGING / SAFETY BLOCK ---
    raw_model_output = build_resnet50_regressor()
    
    # Check if the builder returned a tuple (common in old code versions)
    if isinstance(raw_model_output, tuple):
        print("⚠️ WARNING: build_resnet50_regressor returned a tuple. Unpacking it...")
        model = raw_model_output[0] # Take the first item (the model)
    else:
        model = raw_model_output
        
    print(f"DEBUG: Model type is {type(model)}") # Should be <class 'torchvision.models.resnet.ResNet'>
    # -------------------------------
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    img = Image.open(image_path).convert('RGB')
    transform = get_transforms()
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        
    return output.item()

# try:
#     t = torch.from_numpy(np.array([1, 2, 3]))
#     print("Success: NumPy and PyTorch are talking!")
# except Exception as e:
#     print(f"Still failing: {e}")

#======================= Naive Model ===========================
class NaiveModel: # the naive baseline model that predicts the average age
    def __init__(self, avg_age=50):
        self.avg_age = avg_age
    
    def fit(self, X_train, y_train):
        self.avg_age = y_train.mean()

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        return [self.avg_age] * n_samples

def run_naive_model(X_train, y_train, X_test, y_test):
    naive_model = NaiveModel()
    naive_model.fit(X_train, y_train)
    preds = naive_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    return naive_model, rmse, mae


def run_all_datasets(run_model_func, datasets):
    results = {}
    for modality, (X_train, y_train, _, _, X_test, y_test) in datasets.items():
        model, rmse, mae = run_model_func(X_train, y_train, X_test, y_test)
        with open(f"models/naive_model_{modality}.pkl", "wb") as f:
            pickle.dump(model, f)
        results[modality] = {'RMSE': rmse, 'MAE': mae}
    return results








# ====================== Classical ML pipeline (PCA + ElasticNet) ====================

BEST_PCA_COMPONENTS = 150
BEST_ELASTICNET_ALPHA = 3.0
BEST_ELASTICNET_L1_RATIO = 0.7


def evaluate_rmse_mae(y_true, y_pred):
    """Return RMSE and MAE for regression outputs."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae}


def get_classical_models(
    en_alpha=BEST_ELASTICNET_ALPHA,
    en_l1_ratio=BEST_ELASTICNET_L1_RATIO,
):
    """Return ElasticNet model with configurable hyperparameters."""
    return {
        "ElasticNet": ElasticNet(
            alpha=en_alpha,
            l1_ratio=en_l1_ratio,
            random_state=42,
        ),
    }


def train_and_eval_classical_models(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    models_dict,
):
    """Train models and return metrics on val/test splits (RMSE/MAE)."""
    results = []
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = evaluate_rmse_mae(y_val, val_pred)
        test_metrics = evaluate_rmse_mae(y_test, test_pred)

        results.append({
            "model_name": name,
            "model": model,
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_MAE": test_metrics["MAE"],
        })
    return results


def run_classical_pipeline(
    modality="T1",
    n_components=BEST_PCA_COMPONENTS,
    en_alpha=BEST_ELASTICNET_ALPHA,
    en_l1_ratio=BEST_ELASTICNET_L1_RATIO,
):
    """
    Train ElasticNet on PCA features for one modality.

    Uses precomputed split CSVs under scripts/.
    """
    #repo_root = Path(__file__).resolve().parents[1]
    #module_path = repo_root / "scripts" / "build_features.py"
    #spec = importlib.util.spec_from_file_location("build_features", module_path)
    #build_features = importlib.util.module_from_spec(spec)
    #spec.loader.exec_module(build_features)

    datasets = build_features.build_datasets_from_splits()
    X_train, y_train, X_val, y_val, X_test, y_test = datasets[modality]
    # Suppress numerical warnings from PCA/linear algebra without altering data.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X_train_pca, X_val_pca, X_test_pca, _ = build_features.get_pca_features(
            X_train,
            X_val,
            X_test,
            n_components=n_components,
        )

    models_dict = get_classical_models(
        en_alpha=en_alpha,
        en_l1_ratio=en_l1_ratio,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return train_and_eval_classical_models(
            X_train_pca,
            y_train,
            X_val_pca,
            y_val,
            X_test_pca,
            y_test,
            models_dict,
        )


def run_classical_all_modalities(modalities=None):
    """Run ElasticNet across modalities and return RMSE/MAE per split."""
    if modalities is None:
        modalities = ["T1", "T2", "PD", "MRA"]

    results = {}
    for modality in modalities:
        classical_results = run_classical_pipeline(modality=modality)
        row = classical_results[0]
        with open(f"models/classical_model_{modality}.pkl", "wb") as f:
            pickle.dump(row['model'], f)
        results[modality] = {
            "RMSE": row["test_RMSE"],
            "MAE": row["test_MAE"],
        }
    return results






# ======================== Deep Learning Model (ResNet50) ========================

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class MRIImageDataset(Dataset):
    def __init__(self, image_paths, ages, transform=None):
        self.image_paths = np.array(image_paths)
        self.ages = np.array(ages, dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        age = float(self.ages[idx])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)
    

def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=(5, 9)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            AddGaussianNoise(0.0, 0.03),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return train_transform, val_transform, test_transform

def build_resnet50_regressor(lr=0.001):
    model = resnet50(pretrained=True)

    #for param in model.parameters():
    #    param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, num_workers=2):
    train_transform, val_transform, test_transform = build_transforms()

    train_dataset = MRIImageDataset(X_train, y_train, transform=train_transform)
    val_dataset = MRIImageDataset(X_val, y_val, transform=val_transform)
    test_dataset = MRIImageDataset(X_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def train_model(model, train_loader, val_loader, train_dataset, val_dataset, criterion, optimizer, num_epochs=20, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_losses = []
    val_losses = []
    all_val_preds = []
    all_val_ages = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for images, ages in train_loader:
            images = images.to(device)
            ages = ages.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), ages)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        all_val_preds = []
        all_val_ages = []

        with torch.no_grad():
            for images, ages in val_loader:
                images = images.to(device)
                ages = ages.to(device)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), ages)
                running_val_loss += loss.item() * images.size(0)

                all_val_preds.extend(outputs.squeeze().cpu().numpy().tolist())
                all_val_ages.extend(ages.cpu().numpy().tolist())

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    print("Training complete.")
    return model, train_losses, val_losses, all_val_ages, all_val_preds


def evaluate_regression(all_ages, all_preds):
    mae = mean_absolute_error(all_ages, all_preds)
    mse = mean_squared_error(all_ages, all_preds)
    rmse = float(np.sqrt(mse))
    return float(mae), rmse


def run_training_pipeline(
    *,
    dataset,
    num_epochs=20,
    batch_size=32,
    num_workers=2,
    lr=0.001,
):
    #make_dataset.download_gcs_folder(bucket_name=bucket_name, gcs_folder_prefix=gcs_folder_prefix, local_dir=local_dir)

    #needed_df = load_dataframe(csv_path=csv_path)
    #print(needed_df.shape)

    #needed_df = add_image_paths(needed_df, images_dir=local_dir)

    #X_train, X_val, X_test, y_train, y_val, y_test = split_data(needed_df)

    X_train, y_train, X_val, y_val, X_test, y_test = dataset

    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_val)} samples")
    print(f"Test set size: {len(X_test)} samples")

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size, num_workers=0
    )

    print(f"Training DataLoader created with {len(train_dataset)} samples and batch size {batch_size}.")
    print(f"Validation DataLoader created with {len(val_dataset)} samples and batch size {batch_size}.")
    print(f"Test DataLoader created with {len(test_dataset)} samples and batch size {batch_size}.")

    model, criterion, optimizer = build_resnet50_regressor(lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loss function (MSELoss) and Optimizer (Adam) defined.")
    print("Modified ResNet50 model architecture:")
    print(model)

    model, train_losses, val_losses, all_val_ages, all_val_preds = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    mae, rmse = evaluate_regression(all_val_ages, all_val_preds)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_ages": all_val_ages,
        "val_preds": all_val_preds,
        "MAE": mae,
        "RMSE": rmse,
    }

def run_deep_learning_all_modalities(datasets, num_epochs=20, batch_size=32, num_workers=2, lr=0.001):
    results = {}
    for modality, dataset in datasets.items():
        print(f"Running Deep Learning pipeline for modality: {modality}")
        dl_results = run_training_pipeline(
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
        )

        torch.save(dl_results["model"].state_dict(), f"models/dl_model_{modality}.pth")
        with open(f"models/dl_model_{modality}_full.pkl", "wb") as f:
            pickle.dump(dl_results["model"], f)

        results[modality] = {
            "MAE": dl_results["MAE"],
            "RMSE": dl_results["RMSE"],
        }
    return results


def predict_age_from_file(
    image_path,
    model_weights_path,
    device=None):

    """
    Run inference on a single MRI image file.
    Uses the existing ResNet50 architecture and preprocessing.
    This is inference-only (no training).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model architecture
    model, _, _ = build_resnet50_regressor()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Use test-time transforms only (no augmentation)
    _, _, test_transform = build_transforms()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = test_transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        pred = model(image).item()

    return float(pred)


# REMOVE THIS BEFORE SUBMITTING
if __name__ == "__main__":

    datasets = make_dataset.create_datasets()
    #classical_datasets = build_features.build_datasets_from_splits()
    dl_datasets = make_dataset.create_datasets(type='files')

    # Run the naive baseline model
    naive_results = run_all_datasets(run_naive_model, datasets)
    print("Naive Model Results:")
    for modality, metrics in naive_results.items():
        print(f"\t{modality} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")

    # Run the classical model with the same simple output format
    classical_results = run_classical_all_modalities()
    print("Classical Model Results (PCA features):")
    for modality, metrics in classical_results.items():
        print(f"\t{modality} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")

    # Run the deep learning model with the same simple output format
    deep_learning_results = run_deep_learning_all_modalities(dl_datasets)
    print("Deep Learning Model Results (ResNet50):")
    for modality, metrics in deep_learning_results.items():
        print(f"\t{modality} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")
