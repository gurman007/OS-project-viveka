import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# === Step 1: Load your preprocessed dataset ===
df = pd.read_csv("scripts/smote/preprocessed_for_smote.csv")

# === Step 2: Encode 'max_version' if present ===
if "max_version" in df.columns:
    df["max_version"] = df["max_version"].fillna("unknown")
    le = LabelEncoder()
    df["max_version_encoded"] = le.fit_transform(df["max_version"])
    df.drop(columns=["max_version"], inplace=True)

# === Step 3: Separate features and label ===
X = df.drop(columns=["label"])
y = df["label"]

# === Step 4: Fill missing values (SMOTE can't handle NaNs) ===
X = X.fillna(X.median())

# === Step 5: Apply SMOTE ===
print("🔄 Applying SMOTE to balance dataset...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# === Step 6: Combine and Save ===
balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
balanced_df["label"] = y_resampled

# Save to CSV
balanced_df.to_csv("smote_balanced_dataset.csv", index=False)
print("✅ Done! SMOTE applied successfully.")
print(f"📊 Balanced dataset shape: {balanced_df.shape}")
