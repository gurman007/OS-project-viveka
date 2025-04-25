import pandas as pd
from sklearn.preprocessing import LabelEncoder

# === Load both datasets ===
vuln_df = pd.read_csv("scripts/a_Augmenting/augmented_vulnerable_crates.csv")
safe_df = pd.read_csv("data/top_crates_metadata.csv")

# === Label safe crates ===
safe_df["label"] = 0
safe_df["source"] = "safe_sample"
safe_df["cvss_score"] = 0.0

# === Encode version string ===
if "max_version" in safe_df.columns:
    safe_df["max_version"] = safe_df["max_version"].fillna("unknown")
    le = LabelEncoder()
    safe_df["max_version_encoded"] = le.fit_transform(safe_df["max_version"])
    safe_df.drop(columns=["max_version"], inplace=True)

# === Align column order ===
cols_to_keep = [
    "crate", "downloads", "recent_downloads", "updated_at", "created_at",
    "description", "repository", "cvss_score", "label", "max_version_encoded", "source"
]
vuln_df = vuln_df[cols_to_keep]
safe_df = safe_df[cols_to_keep]

# === Pre-merge counts ===
pre_merge_vuln = len(vuln_df)
pre_merge_safe = len(safe_df)
pre_merge_total = pre_merge_vuln + pre_merge_safe

# === Filter out safe crates that overlap with vulnerable crates ===
safe_df = safe_df[~safe_df["crate"].isin(vuln_df["crate"])]

# === Merge with no risk of overwriting vulnerable entries ===
merged_df = pd.concat([vuln_df, safe_df], ignore_index=True)

# === Save final dataset ===
merged_df.to_csv("final_augmented_training_data.csv", index=False)

# === Output final stats ===
final_total = len(merged_df)
final_vuln = sum(merged_df["label"] == 1)
final_safe = sum(merged_df["label"] == 0)

print(f"✅ Merged and saved to: final_augmented_training_data.csv")
print(f"📦 Crates before deduplication: {pre_merge_total} (Vuln: {pre_merge_vuln}, Safe: {pre_merge_safe})")
print(f"🧹 Final unique crates: {final_total}")
print(f"🐛 Vulnerable: {final_vuln} | 🛡️ Safe: {final_safe}")
