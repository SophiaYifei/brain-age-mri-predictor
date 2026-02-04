import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.api_core\._python_version_support"
)

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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ----------------------------
# Baseline model (kept from your original file)
# ----------------------------
class NaiveModel:  # the naive baseline model that predicts the average age
    def __init__(self, avg_age=50):
        self.avg_age = avg_age

    def fit(self, X_train, y_train):
        self.avg_age = float(np.mean(y_train))

    def predict(self, X_test):
        n_samples = len(X_test)
        return [self.avg_age] * n_samples


def run_naive_model(X_train, y_train, X_test, y_test):
    naive_model = NaiveModel()
    naive_model.fit(X_train, y_train)
    preds = naive_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    return rmse, mae


def run_all_datasets(run_model_func, datasets):
    results = {}
    for modalities, (X_train, y_train, _, _, X_test, y_test) in datasets.items():
        rmse, mae = run_model_func(X_train, y_train, X_test, y_test)
        results[modalities] = {"RMSE": rmse, "MAE": mae}
    return results


# ----------------------------
# Your Colab notebook code, preserved and runnable in model.py
# ----------------------------
def download_gcs_folder(bucket_name, gcs_folder_prefix, local_dir):
    """
    Downloads all blobs from a bucket with a specific prefix.
    """
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)

    if not gcs_folder_prefix.endswith("/"):
        gcs_folder_prefix += "/"

    blobs = bucket.list_blobs(prefix=gcs_folder_prefix)

    print(f"Searching for files in: gs://{bucket_name}/{gcs_folder_prefix}")

    for blob in blobs:
        if blob.name == gcs_folder_prefix:
            continue

        local_file_name = os.path.basename(blob.name)

        if local_file_name:
            local_path = os.path.join(local_dir, local_file_name)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            blob.download_to_filename(local_path)
            print(f"Downloaded: {local_file_name}")


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


# ----------------------------
# Grad-CAM target for regression (fixes RegressionTarget import issues)
# ----------------------------
class RegressionTarget:
    def __call__(self, model_output):
        if model_output.ndim > 1:
            return model_output[:, 0]
        return model_output


def show_gradcam(model, input_tensor, original_image):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [RegressionTarget()]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    visualization = show_cam_on_image(original_image, grayscale_cam[0, :], use_rgb=True)
    return visualization


# ----------------------------
# Main: downloads data, builds paths, splits, trains for epochs, prints losses + metrics
# ----------------------------
if __name__ == "__main__":
    # Remove Colab sample_data equivalent if it exists locally
    if os.path.exists("./sample_data"):
        try:
            import shutil

            shutil.rmtree("./sample_data")
        except Exception:
            pass

    # 1) Download images from GCS
    download_gcs_folder(
        bucket_name="brain-age-mri-bucket",
        gcs_folder_prefix="imgs_folder/",
        local_dir="./data/raw",
    )

    # 2) Load CSV + filter columns
    df = pd.read_csv("./IXI_with_filenames.csv")
    needed_df = df[["IXI_ID", "file_name", "AGE"]].copy()
    needed_df.dropna(subset=["file_name", "AGE"], inplace=True)
    print(needed_df.shape)

    # 3) Create image_path column (same as notebook)
    needed_df = needed_df.copy()
    needed_df["image_path"] = needed_df["file_name"].apply(lambda x: os.path.join("./data/raw", x))

    # 4) Split data (same as notebook)
    X = needed_df["image_path"]
    y = needed_df["AGE"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_val)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # 5) Define transforms (same intent as notebook)
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=(5, 9)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            AddGaussianNoise(0.0, 0.1),
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

    # 6) Create datasets + loaders (same as notebook)
    batch_size = 32
    num_workers = 2

    train_dataset = MRIImageDataset(X_train, y_train, transform=train_transform)
    val_dataset = MRIImageDataset(X_val, y_val, transform=val_transform)
    test_dataset = MRIImageDataset(X_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Training DataLoader created with {len(train_dataset)} samples and batch size {batch_size}.")
    print(f"Validation DataLoader created with {len(val_dataset)} samples and batch size {batch_size}.")
    print(f"Test DataLoader created with {len(test_dataset)} samples and batch size {batch_size}.")

    # 7) Load + modify ResNet50 (same as notebook)
    model = resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    for param in model.fc.parameters():
        param.requires_grad = True

    print("Modified ResNet50 model architecture:")
    print(model)

    # 8) Loss + optimizer (same as notebook)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Loss function (MSELoss) and Optimizer (Adam) defined.")

    # 9) Train loop with epoch prints (same as notebook)
    num_epochs = 20

    train_losses = []
    val_losses = []

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

    # 10) Evaluate (same as notebook intent)
    mae = mean_absolute_error(all_val_ages, all_val_preds)
    mse = mean_squared_error(all_val_ages, all_val_preds)
    rmse = float(np.sqrt(mse))

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")