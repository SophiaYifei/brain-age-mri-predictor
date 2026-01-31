import scripts.make_dataset

def download_data():
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


if __name__ == "__main__":
    #Loads all data from GCS to data/raw directories
    download_data()
    print("Data download complete.")