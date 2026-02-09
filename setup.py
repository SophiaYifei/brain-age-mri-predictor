# No AI used in this script

import scripts.make_dataset
import scripts.model
import scripts.fusion_model
import os

def download_data():
    """Download all modality images from GCS if not already present in data/raw."""
    if not os.path.isdir("data/raw"):
      #Load T1 images
      scripts.make_dataset.download_gcs_folder(
          bucket_name='brain-age-mri-bucket', 
          gcs_folder_prefix='imgs_folder/', 
          local_dir='./data/raw/IXI_T1_png'
      )
      print("T1 images downloaded.")

      #Load T2 images
      scripts.make_dataset.download_gcs_folder(
          bucket_name='brain-age-mri-bucket', 
          gcs_folder_prefix='T2_imgs_folder/', 
          local_dir='./data/raw/IXI_T2_png'
      )
      print("T2 images downloaded.")

      #Load PD images
      scripts.make_dataset.download_gcs_folder(
          bucket_name='brain-age-mri-bucket', 
          gcs_folder_prefix='PD_imgs_folder/', 
          local_dir='./data/raw/IXI_PD_png'
      )
      print("PD images downloaded.")

      #Load MRA images
      scripts.make_dataset.download_gcs_folder(
          bucket_name='brain-age-mri-bucket', 
          gcs_folder_prefix='MRA_imgs_folder/', 
          local_dir='./data/raw/IXI_MRA_png'
      )
      print("MRA images downloaded.")


def train_naive():
    """Run the naive baseline model, save the models, and print results."""
    datasets = scripts.make_dataset.create_datasets()

    # Run the naive baseline model
    naive_results = scripts.model.run_all_datasets(scripts.model.run_naive_model, datasets)
    print("Naive Model Results:")
    for modality, metrics in naive_results.items():
        print(f"\t{modality} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")

def train_classical():
    """Run the classical model with PCA features, save the models, and print results."""
    # Run the classical model with the same simple output format
    classical_results = scripts.model.run_classical_all_modalities()
    print("Classical Model Results (PCA features):")
    for modality, metrics in classical_results.items():
        print(f"\t{modality} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")

def train_dl():
    """Run the deep learning model with ResNet50, save the models, and print results."""
    dl_datasets = scripts.make_dataset.create_datasets(type='files')
    # Run the deep learning model with the same simple output format
    deep_learning_results = scripts.model.run_deep_learning_all_modalities(dl_datasets)
    print("Deep Learning Model Results (ResNet50):")
    for modality, metrics in deep_learning_results.items():
        print(f"\t{modality} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")

def train_dl_fusion():
    """Run the late fusion model, save the model, and print results."""
    scripts.fusion_model.train_fusion()

if __name__ == "__main__":
    #Loads all data from GCS to data/raw directories
    download_data()
    print("Data download complete.")

    #Trains the naive baseline model and prints results
    train_naive()
    print("Naive model training complete and saved for each modality.")

    #Trains the classical model and prints results
    train_classical()
    print("Classical model training complete and saved for each modality.")

    #Trains the deep learning model and prints results
    train_dl()
    print("Deep learning model training complete and saved for each modality.")

    #Trains the late fusion model and prints results
    train_dl_fusion()
    print("Late fusion model training complete and saved.")