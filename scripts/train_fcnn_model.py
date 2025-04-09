import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import random
import numpy as np
#import matplotlib.pyplot as plt

# ----- ⚙️ Configuration -----
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# ----- 📦 Dataset Definition -----
class CrateDataset(Dataset):
    def __init__(self, df, feature_cols, label_col="vulnerable"):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[label_col].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ----- 🔢 Neural Network Definition -----
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ----- 📥 Load Data -----
train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv("data/test_data.csv")

# 🔍 Feature columns
feature_cols = ["downloads", "recent_downloads"]

# Normalize features
for col in feature_cols:
    max_val = train_df[col].max()
    train_df[col] = train_df[col] / max_val
    test_df[col] = test_df[col] / max_val

# ----- 🧾 Datasets and Loaders -----
train_dataset = CrateDataset(train_df, feature_cols)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- 🧠 Initialize Model -----
model = FCNN(input_dim=len(feature_cols))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----- 🚀 Training Loop -----
print("🚀 Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # 🧪 Random 1-sample validation
    model.eval()
    sample = test_df.sample(n=1)
    X_sample = torch.tensor(sample[feature_cols].values.astype(np.float32))
    y_true = sample["vulnerable"].values.astype(int)

    with torch.no_grad():
        y_pred = model(X_sample).squeeze().numpy()
        y_pred_label = int(y_pred > 0.5)
        loss_val = criterion(torch.tensor(y_pred).float(), torch.tensor(float(y_true[0]))).item()
        f1 = f1_score(y_true, [y_pred_label], zero_division=1)

    print(f"📘 Epoch {epoch+1}/{EPOCHS} | 🧠 Train Loss: {avg_train_loss:.4f} | 🧪 Val Loss (1-sample): {loss_val:.4f} | F1-score: {f1:.4f}")

print("✅ Training Complete!")

# ----- 📊 Final Evaluation on Full Test Set -----
print("\n🔍 Final Evaluation on Entire Test Set:")
model.eval()
X_all = torch.tensor(test_df[feature_cols].values.astype(np.float32))
y_true = test_df["vulnerable"].values.astype(int)

with torch.no_grad():
    y_probs = model(X_all).squeeze().numpy()
    y_preds = (y_probs > 0.5).astype(int)

# Show results
print("📌 Confusion Matrix:")
print(confusion_matrix(y_true, y_preds))
print("\n📊 Classification Report:")
print(classification_report(y_true, y_preds, zero_division=1))

# # ----- 💾 Optional: Save the model -----
# torch.save(model.state_dict(), "models/fcnn_model.pth")
# print("✅ Model saved to models/fcnn_model.pth")
