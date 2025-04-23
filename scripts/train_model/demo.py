import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datetime import datetime

# =============================
# 1. LOAD & PREPROCESS DATASET
# =============================
df = pd.read_csv("final_augmented_training_data.csv")

# Drop non-numeric text columns
df = df.drop(columns=["crate", "repository", "description", "repo_slug", "source"], errors="ignore")

# Encode version strings
if "max_version" in df.columns:
    df["max_version"] = df["max_version"].fillna("unknown")
    le = LabelEncoder()
    df["max_version_encoded"] = le.fit_transform(df["max_version"])
    df.drop(columns=["max_version"], inplace=True)

# Convert date fields to "days ago"
def to_days_ago(x):
    try:
        return (datetime.utcnow() - pd.to_datetime(x)).days
    except:
        return np.nan

for col in ["created_at", "updated_at", "last_pushed"]:
    if col in df.columns:
        df[col] = df[col].apply(to_days_ago)
        df.rename(columns={col: col + "_days_ago"}, inplace=True)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

# ======================
# 2. TRAIN / TEST SPLIT
# ======================
X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# 3. DATASET & DATALOADER
# ======================
class CrateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CrateDataset(X_train, y_train)
test_dataset = CrateDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ======================
# 4. DEFINE FCNN MODEL
# ======================
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)  # No Sigmoid here!
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCNN(X_train.shape[1]).to(device)

# ======================
# 5. TRAIN THE MODEL
# ======================
# ======================
# 5. TRAIN THE MODEL with Class Weights
# ======================

# Calculate class weight for positive class (label = 1)
# Formula: pos_weight = total_negative / total_positive
total_pos = np.sum(y_train)
total_neg = len(y_train) - total_pos
pos_weight_value = total_neg / total_pos

pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"📘 Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f}")


# ======================
# 6. EVALUATE ON TEST SET
# ======================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int).flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))
print("🧱 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
