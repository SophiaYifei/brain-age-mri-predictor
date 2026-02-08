import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import scripts.model as model_module

class MultiModalBrainDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        # Column names for your modalities
        self.modalities = ['T1', 'T2', 'PD', 'MRA']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
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
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class LateFusionBrainAgeModel(nn.Module):
    def __init__(self):
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
    
