import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# === Step 1: Load your dataset ===
df = pd.read_csv("final_augmented_training_data.csv")

# === Step 2: Drop non-numeric or non-useful columns ===
drop_cols = ["crate", "repository", "description", "source"]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# === Step 3: Encode max_version if string still present ===
if "max_version" in df.columns:
    df["max_version"] = df["max_version"].fillna("unknown")
    le = LabelEncoder()
    df["max_version_encoded"] = le.fit_transform(df["max_version"])
    df.drop(columns=["max_version"], inplace=True)

# === Step 4: Convert date fields to "days ago" ===
def to_days_ago(x):
    try:
        return (datetime.utcnow() - pd.to_datetime(x)).days
    except:
        return None

for col in ["created_at", "updated_at", "last_pushed"]:
    if col in df.columns:
        df[col] = df[col].apply(to_days_ago)
        df.rename(columns={col: col + "_days_ago"}, inplace=True)

# === Step 5: Ensure all data is numeric + fill NaNs ===
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(df.median())

# === Step 6: Save clean ML-ready dataset ===
df.to_csv("preprocessed_for_training.csv", index=False)
print("✅ Saved: preprocessed_for_training.csv")
print(f"📊 Shape: {df.shape} | Columns: {list(df.columns)}")
