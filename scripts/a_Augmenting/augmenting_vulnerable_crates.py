import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# === Load your vulnerable crates dataset ===
df = pd.read_csv("fully_enriched_vulnerable_dataset.csv")

# === Encode version if present ===
if "max_version" in df.columns:
    df["max_version"] = df["max_version"].fillna("unknown")
    le = LabelEncoder()
    df["max_version_encoded"] = le.fit_transform(df["max_version"])
    df.drop(columns=["max_version"], inplace=True)

# === Columns to apply jitter (small random changes) ===
jitter_cols = [
    "downloads", "recent_downloads", "stars", "forks",
    "open_issues", "cvss_score", "created_at", "updated_at", "last_pushed"
]
jitter_cols = [col for col in jitter_cols if col in df.columns]

# Ensure jitter columns are numeric
for col in jitter_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # convert to float, NaNs if failed
        df[col] = df[col].fillna(df[col].median())         # fill NaNs with median

# === Create 1200 synthetic crates ===
np.random.seed(42)
augmented = []

for _ in range(1200):
    base = df.sample(1).iloc[0].copy()
    new = base.copy()

    for col in jitter_cols:
        if pd.notna(base[col]):
            pct = 0.15 if "downloads" in col or "stars" in col else 0.10
            noise = np.random.uniform(-pct, pct)
            new[col] = max(0, base[col] * (1 + noise))

    if "max_version_encoded" in new:
        new["max_version_encoded"] = max(0, new["max_version_encoded"] + np.random.choice([-1, 0, 1]))

    new["source"] = "augmented"
    new["label"] = 1
    augmented.append(new)

# === Combine and save ===
augmented_df = pd.DataFrame(augmented)
augmented_df.to_csv("scripts/a_Augmenting/augmented_vulnerable_crates.csv", index=False)

print("✅ 1200 synthetic vulnerable crates saved to 'augmented_vulnerable_crates.csv'")
