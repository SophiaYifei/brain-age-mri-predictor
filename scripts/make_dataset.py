import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
import numpy as np
import PIL.Image as Image

def download_gcs_folder(bucket_name, gcs_folder_prefix, local_dir):
    """
    Downloads all blobs from a bucket with a specific prefix.
    """
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)

    # The prefix filters the 'folder'
    # Ensure it ends with a '/' 
    if not gcs_folder_prefix.endswith('/'):
        gcs_folder_prefix += '/'

    blobs = bucket.list_blobs(prefix=gcs_folder_prefix)

    print(f"Searching for files in: gs://{bucket_name}/{gcs_folder_prefix}")

    for blob in blobs:
        # 1. Skip if the blob is just the folder placeholder itself 
        if blob.name == gcs_folder_prefix:
            continue

        # 2. Extract only the filename (remove the folder prefix)
        # If blob.name is 'imgs_folder/IXI050.png', 
        # local_file_name becomes 'IXI050.png'
        local_file_name = os.path.basename(blob.name)
        
        if local_file_name: # Ensure it's not a sub-directory
            local_path = os.path.join(local_dir, local_file_name)
            
            # 3. Create local folder if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # 4. Download
            blob.download_to_filename(local_path)
            #print(f"Downloaded: {local_file_name}")

def load_image_as_array(file_path, target_size=(224, 224)):
    # Load the image, convert to grayscale (L) or RGB (RGB) and all standard size
    with Image.open(file_path) as img:
        img = img.convert('L')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)

def create_datasets(type='files'):
    df = pd.read_csv('../IXI_with_filenames.csv')
    modalities = ['T1', 'T2', 'PD', 'MRA']

    datasets = {}

    print(len(df), "total records in CSV")
    df = df.dropna(subset=['T1_file_name', 'T2_file_name', 'PD_file_name', 'MRA_file_name', 'AGE'])
    df = df.drop_duplicates(subset=['IXI_ID'], keep='first')
    print(len(df), "records after dropping missing and duplicate filenames/ages")

    ids = df['IXI_ID']

    train_val_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.25, random_state=42)

    train_df = df[df['IXI_ID'].isin(train_ids)].copy()
    val_df = df[df['IXI_ID'].isin(val_ids)].copy()
    test_df = df[df['IXI_ID'].isin(test_ids)].copy()

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # for future use, save the splits
    train_df.to_csv('../data/labels/train_split.csv', index=False)
    val_df.to_csv('../data/labels/val_split.csv', index=False)
    test_df.to_csv('../data/labels/test_split.csv', index=False)

    for mod in modalities:
        col_name = f'{mod}_file_name'

        dirname = 'raw'
        #if mod == 'T1':
        #    dirname = 'processed'
        
        if type == 'files':
            # Just return filenames
            X_train = np.array([os.path.join(f'../data/{dirname}/IXI_{mod}_png', fname) for fname in train_df[col_name]])
            y_train = train_df['AGE'].values
            X_val = np.array([os.path.join(f'../data/{dirname}/IXI_{mod}_png', fname) for fname in val_df[col_name]])
            y_val = val_df['AGE'].values
            X_test = np.array([os.path.join(f'../data/{dirname}/IXI_{mod}_png', fname) for fname in test_df[col_name]])
            y_test = test_df['AGE'].values

            datasets[mod] = (X_train, y_train, X_val, y_val, X_test, y_test)
            
        else:
            X_train = np.array([load_image_as_array(os.path.join(f'../data/{dirname}/IXI_{mod}_png', fname)) for fname in train_df[col_name]])
            y_train = train_df['AGE'].values
            X_val = np.array([load_image_as_array(os.path.join(f'../data/{dirname}/IXI_{mod}_png', fname)) for fname in val_df[col_name]])
            y_val = val_df['AGE'].values
            X_test = np.array([load_image_as_array(os.path.join(f'../data/{dirname}/IXI_{mod}_png', fname)) for fname in test_df[col_name]])
            y_test = test_df['AGE'].values

            datasets[mod] = (X_train, y_train, X_val, y_val, X_test, y_test)
    
    return datasets