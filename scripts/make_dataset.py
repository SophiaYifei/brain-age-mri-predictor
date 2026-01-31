import os
from google.cloud import storage

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
