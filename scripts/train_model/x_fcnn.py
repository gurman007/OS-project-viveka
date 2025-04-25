import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

# === Step 1: Load preprocessed data ===
df = pd.read_csv("preprocessed_for_training.csv")
X = df.drop(columns=["label"])
y = df["label"]

# === Step 2: Split and scale ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Step 3: Convert to tensors ===
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

# === Step 4: Define neural network ===
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

input_dim = X_train_tensor.shape[1]
model = FCNN(input_dim)

# === Step 5: Loss & optimizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
pos_weight = torch.tensor([y_train_tensor.shape[0] / y_train_tensor.sum()], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Step 6: Training loop ===
for epoch in range(25):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/25 - Loss: {total_loss:.4f}")

# === Step 7: Evaluation ===
model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_tensor.to(device))
    y_pred = (torch.sigmoid(y_pred_logits) > 0.5).cpu().numpy()

print("🧱 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
