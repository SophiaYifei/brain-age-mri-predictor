import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

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



#REMOVE THIS BEFORE SUBMITTING
if __name__ == "__main__":
    datasets = make_dataset.create_datasets()

    # Run the naive baseline model
    naive_results = run_all_datasets(run_naive_model, datasets)
    print("Naive Model Results:")
    for modality, metrics in naive_results.items():
        print(f"\t{modality} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")

