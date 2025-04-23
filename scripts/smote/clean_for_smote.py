import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# === Step 1: Load original enriched dataset ===
df = pd.read_csv("full_final_labeled_dataset.csv")

# === Step 2: Drop non-numeric or non-useful columns ===
columns_to_drop = ["crate", "repository", "description", "repo_slug", "source"]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# === Step 3: Convert version string to numeric (label encode) ===
if "max_version" in df.columns:
    df["max_version"] = df["max_version"].fillna("unknown")
    le = LabelEncoder()
    df["max_version_encoded"] = le.fit_transform(df["max_version"])
    df.drop(columns=["max_version"], inplace=True)

# === Step 4: Convert dates to 'days ago' ===
def to_days_ago(x):
    try:
        return (datetime.utcnow() - pd.to_datetime(x)).days
    except:
        return None

for col in ["created_at", "updated_at", "last_pushed"]:
    if col in df.columns:
        df[col] = df[col].apply(to_days_ago)
        df.rename(columns={col: f"{col}_days_ago"}, inplace=True)

# === Step 5: Force all columns to numeric + Fill NaNs ===
df = df.apply(pd.to_numeric, errors="coerce")  # force numeric, turn junk into NaN
df = df.fillna(df.median())                    # fill NaNs with median

# === Step 6: Save clean file for SMOTE ===
df.to_csv("scripts/smote/preprocessed_for_smote.csv", index=False)
print("✅ File saved: preprocessed_for_smote.csv (100% SMOTE-ready)")
print(f"📊 Shape: {df.shape}")
