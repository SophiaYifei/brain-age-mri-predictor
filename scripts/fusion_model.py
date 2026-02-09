# AI used: Gemini 3 https://gemini.google.com/share/774e5541237c

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import scripts.model as model_module


# --- Path Setup ---
# Calculates the path to the 'models' folder relative to this script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
# This filename must match what download_weights.py saves
PATH_FUSION = os.path.join(MODELS_DIR, "final_late_fusion_model.pth")


# --- Helper Functions ---

def get_transforms():
    """
    Standard preprocessing for ResNet.
    Matches the 'test_transform' used in training.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# --- Inference Function ---

def predict_fusion_model(image_paths_dict, weights_path=None):
    """
    Runs inference using the LateFusionBrainAgeModel.
    
    Args:
        image_paths_dict (dict): Dictionary with keys 'T1', 'T2', 'PD', 'MRA' pointing to image files.
        weights_path (str, optional): Path to .pth file. Defaults to standard location.
    
    Returns:
        float: Predicted age
    """
    if weights_path is None:
        weights_path = PATH_FUSION

    # Sanity Check
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. "
            "Did you run 'python scripts/download_weights.py'?"
        )
        
    device = torch.device("cpu") # Use CPU for inference deployments
    
    # 1. Initialize the exact architecture
    model = LateFusionBrainAgeModel() 
    
    # 2. Load Weights
    # map_location='cpu' is required if model was trained on GPU
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except RuntimeError as e:
        # Fallback: Sometimes DataParallel wraps keys with "module."
        # This handles that case automatically
        print("Standard load failed, attempting to fix key mismatch...")
        state_dict = torch.load(weights_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()
    
    # 3. Preprocess Inputs
    transform = get_transforms()
    tensors = []
    
    try:
        # Order matters! Must match forward(x1, x2, x3, x4) -> T1, T2, PD, MRA
        for mod in ['T1', 'T2', 'PD', 'MRA']:
            if mod not in image_paths_dict:
                 raise ValueError(f"Missing image path for {mod}")
                 
            img = Image.open(image_paths_dict[mod]).convert('RGB')
            # Add batch dimension: [1, 3, 224, 224]
            tensors.append(transform(img).unsqueeze(0).to(device))
            
    except Exception as e:
        raise ValueError(f"Error processing images: {e}")
        
    # 4. Predict
    with torch.no_grad():
        # Unpack the list of 4 tensors into the forward method
        output = model(*tensors)
        
    return output.item()


class MultiModalBrainDataset(Dataset):
    """
    MultiModalBrainDataset brings together all 4 modalities for a single subject as one sample.
    """
    def __init__(self, df, root_dir, transform=None):
        """ Standard PyTorch Dataset init. Expects a DataFrame with columns for each modality's filename and the target age. """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        # Column names for your modalities
        self.modalities = ['T1', 'T2', 'PD', 'MRA']

    def __len__(self):
        """ Length is just the number of rows in the DataFrame, since each row corresponds to one subject with all modalities. """
        return len(self.df)

    def __getitem__(self, idx):
        """ Gets one sample: all 4 modality images + age for the subject at index idx. """
        row = self.df.iloc[idx]
        images = []
        
        for mod in self.modalities:
            # Construct path: ../data/raw/IXI_T1_png/filename.png
            img_path = os.path.join(self.root_dir, f"IXI_{mod}_png", row[f'{mod}_file_name'])
            img = Image.open(img_path).convert('RGB') # ResNet expects 3 channels
            
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        age = torch.tensor(row['AGE'], dtype=torch.float32)
        
        # Returns: (T1, T2, PD, MRA), Age
        return images[0], images[1], images[2], images[3], age

class AddGaussianNoise(object):
    """ Data augmentation: Adds random Gaussian noise to the input tensor. Useful for regularization and improving robustness. """
    def __init__(self, mean=0.0, std=1.0):
        """ Initializes the noise parameters. Mean is typically 0, and std controls the noise intensity. """
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """ Adds Gaussian noise to the input tensor. The noise is generated with the specified mean and std, and added to the original tensor. """
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        """ String representation for debugging. Shows the mean and std of the noise. """
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class LateFusionBrainAgeModel(nn.Module):
    """ Brain age prediction model using late fusion. Each modality has its own "expert" branch (ResNet50), and their outputs are combined in a "fusion head" to predict age. """
    def __init__(self):
        """ Initializes the model architecture. We create 4 ResNet50 branches for the 4 modalities, remove their final fully connected layers to get feature vectors, and then define a fusion head that takes all features and outputs a single age prediction. """
        super(LateFusionBrainAgeModel, self).__init__()
        
        # 1. We create 4 "Expert" branches
        # We use resnet50 because that's what you trained your .pth files on
        self.branch_t1  = models.resnet50(weights=None)
        self.branch_t2  = models.resnet50(weights=None)
        self.branch_pd  = models.resnet50(weights=None)
        self.branch_mra = models.resnet50(weights=None)
        
        # 2. Network Surgery: Replace the 'fc' (fully connected) layer with Identity
        # This stops the branches from predicting age and makes them output 
        # a "feature vector" (2048 numbers) instead.
        num_features = self.branch_t1.fc.in_features # 2048 for ResNet50
        
        self.branch_t1.fc  = nn.Identity()
        self.branch_t2.fc  = nn.Identity()
        self.branch_pd.fc  = nn.Identity()
        self.branch_mra.fc = nn.Identity()
        
        # 3. The Fusion Head (The "Judge")
        # This takes the 2048 features from each of the 4 branches (total 8192)
        # and learns how to combine them into one final age.
        self.regressor = nn.Sequential(
            nn.Linear(num_features * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # The final predicted Age
        )

    def forward(self, x1, x2, x3, x4):
        """ Defines the forward pass. Each input corresponds to one modality (T1, T2, PD, MRA). Each modality goes through its own branch to extract features, and then all features are concatenated and passed through the fusion head to get the final age prediction. """
        # Pass each modality through its own expert branch
        feat1 = self.branch_t1(x1)
        feat2 = self.branch_t2(x2)
        feat3 = self.branch_pd(x3)
        feat4 = self.branch_mra(x4)
        
        # Combine the "opinions" of all 4 branches into one long vector
        combined = torch.cat((feat1, feat2, feat3, feat4), dim=1)
        
        # Predict the age based on the combined information
        return self.regressor(combined)


def build_fusion_model(model_paths, device):
    """ Builds the LateFusionBrainAgeModel and loads the pretrained weights for each branch. Initially, all branches are frozen and only the fusion head is trained. After loading weights, the model is moved to the specified device (CPU or GPU). """
    model = LateFusionBrainAgeModel().to(device)
    
    # Mapping model branches to your saved files
    mapping = {
        model.branch_t1: model_paths['T1'],
        model.branch_t2: model_paths['T2'],
        model.branch_pd: model_paths['PD'],
        model.branch_mra: model_paths['MRA']
    }
    
    for branch, path in mapping.items():
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=device)
            # strict=False because we replaced the 'fc' layer with Identity
            branch.load_state_dict(state_dict, strict=False)
            
            # Initially freeze branches
            for param in branch.parameters():
                param.requires_grad = False
        else:
            print(f"Warning: {path} not found. Branch will be random.")
            
    return model

def validate(model, loader, criterion, device):
    """ Validates the model on the validation set. This is a standard evaluation loop that runs the model in inference mode (no gradients) and calculates the average loss across the validation dataset. """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation to save memory
        for t1, t2, pdi, mra, age in loader:
            t1, t2, pdi, mra, age = t1.to(device), t2.to(device), pdi.to(device), mra.to(device), age.to(device)
            output = model(t1, t2, pdi, mra)
            loss = criterion(output.squeeze(), age)
            val_loss += loss.item()
    return val_loss / len(loader)


def train_fusion():
    """ Trains the LateFusionBrainAgeModel in two stages: first only the fusion head is trained while the branches are frozen, and then the entire network is fine-tuned. The best model based on validation loss is saved to disk. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize
    model_paths = {
        'T1': 'models/dl_model_T1.pth', 'T2': 'models/dl_model_T2.pth',
        'PD': 'models/dl_model_PD.pth', 'MRA': 'models/dl_model_MRA.pth'
    }
    model = build_fusion_model(model_paths, device)

    data_transforms = model_module.build_transforms()
    
    train_df = pd.read_csv('data/labels/train_split.csv')
    val_df = pd.read_csv('data/labels/val_split.csv')
    test_df = pd.read_csv('data/labels/test_split.csv')

    # 2. Data Loaders
    train_ds = MultiModalBrainDataset(train_df, "data/raw", transform=data_transforms[0])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_ds = MultiModalBrainDataset(val_df, "data/raw", transform=data_transforms[1])
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_ds = MultiModalBrainDataset(test_df, "data/raw", transform=data_transforms[2])
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # 3. Optimization
    criterion = torch.nn.L1Loss() # MAE is best for brain age
    optimizer = torch.optim.Adam(model.regressor.parameters(), lr=1e-3)

    best_val_loss = float('inf')

    # --- STAGE 1: Train Head Only ---
    print("Starting Stage 1: Training Fusion Head...")
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for t1, t2, pdi, mra, age in train_loader:
            t1, t2, pdi, mra, age = t1.to(device), t2.to(device), pdi.to(device), mra.to(device), age.to(device)
            optimizer.zero_grad()
            output = model(t1, t2, pdi, mra)
            loss = criterion(output.squeeze(), age)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch} | Train MAE: {avg_train_loss:.2f} | Val MAE: {avg_val_loss:.2f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/final_late_fusion_model.pth')
            print("--> Best Model Saved (Stage 1)")

    # --- STAGE 2: Fine-Tuning ---
    print("\nStarting Stage 2: Fine-tuning entire network...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for t1, t2, pdi, mra, age in train_loader:
            t1, t2, pdi, mra, age = t1.to(device), t2.to(device), pdi.to(device), mra.to(device), age.to(device)
            optimizer.zero_grad()
            output = model(t1, t2, pdi, mra)
            loss = criterion(output.squeeze(), age)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Fine-Tune Epoch {epoch} | Train MAE: {avg_train_loss:.2f} | Val MAE: {avg_val_loss:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/final_late_fusion_model.pth')
            print("--> Best Model Saved (Stage 2)")

    # Save the final weights after all epochs
    torch.save(model.state_dict(), 'models/final_late_fusion_model.pth')
    print("Final model saved as final_late_fusion_model.pth")

    # Optional: Save the entire model object (architecture + weights)
    # This is useful if you don't want to redefine the class later
    torch.save(model, 'models/full_fusion_model_object.pt')
    
