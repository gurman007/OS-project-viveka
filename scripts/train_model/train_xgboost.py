import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ⚙️ Configuration
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# -------------------------------
# Step 1: Load and Engineer Features
# -------------------------------
df = pd.read_csv("final_augmented_training_data.csv")
print(f"📦 Loaded {len(df)} crates.")

# Make current time timezone-aware
now = pd.Timestamp.now(tz="UTC")

# Convert datetime fields to timezone-aware datetimes
df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

# Derived datetime features
df["days_since_update"] = (now - df["updated_at"]).dt.days
df["days_since_created"] = (now - df["created_at"]).dt.days

# Parse version into parts
version_split = df["max_version"].fillna("0.0.0").str.extract(r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")
df["version_major"] = version_split["major"].astype(float)
df["version_minor"] = version_split["minor"].astype(float)
df["version_patch"] = version_split["patch"].astype(float)

# Derived feature: description length
df["desc_length"] = df["description"].fillna("").apply(len)

# Derived feature: binary flag for repository presence
df["has_repository"] = df["repository"].notna().astype(int)

# Define the features we want to use
feature_cols = [
    "downloads", "recent_downloads",
    "days_since_update", "days_since_created",
    "version_major", "version_minor", "version_patch",
    "desc_length", "has_repository"
]

# Remove rows with missing critical features and labels
df = df.dropna(subset=feature_cols + ["vulnerable"]).reset_index(drop=True)

# Normalize numerical features
for col in feature_cols:
    max_val = df[col].max()
    df[col] = df[col] / max_val

# -------------------------------
# Step 2: Split into Train/Test and Upsample the Minority Class
# -------------------------------
train_df, test_df = train_test_split(df, test_size=0.05, random_state=42, shuffle=True)
print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")

# Upsampling the minority class in the training set
# Separate safe (vulnerable = 0) and risky (vulnerable = 1)
train_safe = train_df[train_df["vulnerable"] == 0]
train_risky = train_df[train_df["vulnerable"] == 1]
print(f"Before upsampling: safe = {len(train_safe)}, risky = {len(train_risky)}")

# Upsample risky samples: replicate with replacement to match safe count
if len(train_risky) > 0:
    train_risky_upsampled = resample(train_risky, replace=True, n_samples=len(train_safe), random_state=42)
    train_df_balanced = pd.concat([train_safe, train_risky_upsampled])
    # Shuffle the combined dataset
    train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"After upsampling: Total training samples = {len(train_df_balanced)}")
else:
    print("Warning: No risky examples in training! Using original training set.")
    train_df_balanced = train_df

# -------------------------------
# Step 3: Create PyTorch Dataset and Loader
# -------------------------------
class CrateDataset(Dataset):
    def __init__(self, dataframe, feature_cols):
        self.X = dataframe[feature_cols].values.astype(np.float32)
        self.y = dataframe["vulnerable"].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

train_dataset = CrateDataset(train_df_balanced, feature_cols)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Step 4: Define the Neural Network Model
# -------------------------------
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 40),
            nn.ReLU(),
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability for binary classification
        )
        
    def forward(self, x):
        return self.model(x)

model = FCNN(input_dim=len(feature_cols))
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------
# Step 5: Training Loop with Intermediate Full Test Evaluation
# -------------------------------
print("🚀 Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    
    # 1-sample validation (for quick debug)
    model.eval()
    sample = test_df.sample(n=1)
    X_sample = torch.tensor(sample[feature_cols].values.astype(np.float32))
    y_true_sample = sample["vulnerable"].values.astype(int)
    with torch.no_grad():
        y_pred_sample = model(X_sample).squeeze().numpy()
        y_pred_label_sample = int(y_pred_sample > 0.5)
        f1_debug = f1_score(y_true_sample, [y_pred_label_sample], zero_division=1)
        val_loss_sample = criterion(torch.tensor(y_pred_sample).float(), torch.tensor(float(y_true_sample[0]))).item()
    
    print(f"📘 Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss (1-sample): {val_loss_sample:.4f} | Classification (sample): {f1_debug:.4f}")
    
    # Full test set evaluation every 5 epochs or in final epoch
    if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
        X_test = torch.tensor(test_df[feature_cols].values.astype(np.float32))
        y_test = test_df["vulnerable"].values.astype(int)
        with torch.no_grad():
            y_prob = model(X_test).squeeze().numpy()
            y_pred_full = (y_prob > 0.5).astype(int)
        print("\n🔍 Full Test Evaluation:")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_full))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_full, zero_division=1))
        print("────────────────────────────\n")
