from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_MODALITIES = ("T1", "T2", "PD", "MRA")


def get_repo_root() -> Path:
    """Return the repository root (parent of scripts/)."""
    return Path(__file__).resolve().parents[1]


def load_make_dataset_module(repo_root: Path):
    """Load scripts/make_dataset.py without modifying sys.path."""
    module_path = repo_root / "scripts" / "make_dataset.py"
    # Load dynamically to keep this module self-contained.
    spec = importlib.util.spec_from_file_location("make_dataset", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split_csvs(repo_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits stored in scripts/."""
    train_df = pd.read_csv(repo_root / "scripts" / "train_split.csv")
    val_df = pd.read_csv(repo_root / "scripts" / "val_split.csv")
    test_df = pd.read_csv(repo_root / "scripts" / "test_split.csv")
    return train_df, val_df, test_df


def load_split_arrays(
    df: pd.DataFrame,
    modality: str,
    repo_root: Path,
    make_dataset_module,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load images and labels for one split + modality."""
    # Map the modality to its filename column and image folder.
    col_name = f"{modality}_file_name"
    img_dir = repo_root / "data" / "raw" / f"IXI_{modality}_png"
    # Load images as arrays using the shared helper in make_dataset.py.
    X = np.array([
        make_dataset_module.load_image_as_array(os.path.join(img_dir, fname))
        for fname in df[col_name]
    ])
    y = df["AGE"].values
    return X, y


def build_datasets_from_splits(
    modalities: Iterable[str] = DEFAULT_MODALITIES,
    repo_root: Path | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build datasets dict from precomputed split CSVs.

    Returns:
        datasets[modality] = (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    repo_root = repo_root or get_repo_root()
    train_df, val_df, test_df = load_split_csvs(repo_root)
    make_dataset_module = load_make_dataset_module(repo_root)

    # Build a dict keyed by modality with train/val/test arrays.
    datasets = {}
    for mod in modalities:
        X_train, y_train = load_split_arrays(train_df, mod, repo_root, make_dataset_module)
        X_val, y_val = load_split_arrays(val_df, mod, repo_root, make_dataset_module)
        X_test, y_test = load_split_arrays(test_df, mod, repo_root, make_dataset_module)
        datasets[mod] = (X_train, y_train, X_val, y_val, X_test, y_test)
    return datasets


def flatten_images(X: np.ndarray) -> np.ndarray:
    """Flatten 2D images into vectors."""
    return X.reshape(X.shape[0], -1)


def build_pca_pipeline(n_components: int) -> Pipeline:
    """Create a scaler + PCA pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])


def get_pca_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 150,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """Fit PCA on train split and transform train/val/test."""
    # Flatten images before PCA since PCA expects 2D [n_samples, n_features].
    X_train_flat = flatten_images(X_train)
    X_val_flat = flatten_images(X_val)
    X_test_flat = flatten_images(X_test)

    # Fit PCA only on training data to avoid leakage.
    pca_pipe = build_pca_pipeline(n_components)
    X_train_pca = pca_pipe.fit_transform(X_train_flat)
    X_val_pca = pca_pipe.transform(X_val_flat)
    X_test_pca = pca_pipe.transform(X_test_flat)

    return X_train_pca, X_val_pca, X_test_pca, pca_pipe


def concat_features(*arrays: np.ndarray) -> np.ndarray:
    """Concatenate multiple feature matrices along feature axis."""
    return np.concatenate(arrays, axis=1)
