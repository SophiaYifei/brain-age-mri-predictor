import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import RegressionTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import make_dataset

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
    return rmse, mae


def run_all_datasets(run_model_func, datasets):
    results = {}
    for modalities, (X_train, y_train, _, _, X_test, y_test) in datasets.items():
        rmse, mae = run_model_func(X_train, y_train, X_test, y_test)
        results[modalities] = {'RMSE': rmse, 'MAE': mae}
    return results





def show_gradcam(model, input_tensor, original_image):
    # 1. Define the layer you want to look at 
    # For ResNet50, it's usually the last block
    target_layers = [model.layer4[-1]]

    # 2. Initialize GradCAM for Regression
    cam = GradCAM(model=model, target_layers=target_layers)

    # 3. Create a target (since we want the actual age value)
    # If the model predicts '65', we want to see what led to that '65'
    targets = [RegressionTarget()]

    # 4. Generate the heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # 5. Overlay it on your original MRI slice
    visualization = show_cam_on_image(original_image, grayscale_cam[0, :], use_rgb=True)



# --- Classical ML pipeline (PCA + Ridge/ElasticNet) ---

BEST_PCA_COMPONENTS = 150
BEST_ELASTICNET_ALPHA = 3.0
BEST_ELASTICNET_L1_RATIO = 0.7


def evaluate_rmse_mae(y_true, y_pred):
    """Return RMSE and MAE for regression outputs."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae}


def get_classical_models(
    ridge_alpha=1.0,
    en_alpha=BEST_ELASTICNET_ALPHA,
    en_l1_ratio=BEST_ELASTICNET_L1_RATIO,
):
    """Return Ridge and ElasticNet models with configurable hyperparameters."""
    return {
        "Ridge": Ridge(alpha=ridge_alpha),
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
            "model": name,
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_MAE": test_metrics["MAE"],
        })
    return results


def run_classical_pipeline(
    modality="T1",
    n_components=BEST_PCA_COMPONENTS,
    ridge_alpha=1.0,
    en_alpha=BEST_ELASTICNET_ALPHA,
    en_l1_ratio=BEST_ELASTICNET_L1_RATIO,
):
    """
    Train Ridge/ElasticNet on PCA features for one modality.

    Uses precomputed split CSVs under scripts/.
    """
    from scripts import build_features

    datasets = build_features.build_datasets_from_splits()
    X_train, y_train, X_val, y_val, X_test, y_test = datasets[modality]
    X_train_pca, X_val_pca, X_test_pca, _ = build_features.get_pca_features(
        X_train,
        X_val,
        X_test,
        n_components=n_components,
    )

    models_dict = get_classical_models(
        ridge_alpha=ridge_alpha,
        en_alpha=en_alpha,
        en_l1_ratio=en_l1_ratio,
    )
    return train_and_eval_classical_models(
        X_train_pca,
        y_train,
        X_val_pca,
        y_val,
        X_test_pca,
        y_test,
        models_dict,
    )


#REMOVE THIS BEFORE SUBMITTING
if __name__ == "__main__":
    datasets = make_dataset.create_datasets()

    # Run the naive baseline model
    naive_results = run_all_datasets(run_naive_model, datasets)
    print("Naive Model Results:")
    for modality, metrics in naive_results.items():
        print(f"\t{modality} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")

    classical_results = run_classical_pipeline()
    print("Classical Model Results (PCA features):")
    for row in classical_results:
        print(row)