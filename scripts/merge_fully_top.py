import pandas as pd

# Load datasets
vuln_df = pd.read_csv("fully_enriched_vulnerable_dataset.csv")
safe_df = pd.read_csv("data/top_crates_metadata.csv")

# Normalize crate names for matching
vuln_df["crate_std"] = vuln_df["crate"].str.strip().str.lower()
safe_df["crate_std"] = safe_df["crate"].str.strip().str.lower()

# Remove overlapping crates (already in vulnerable list)
safe_cleaned = safe_df[~safe_df["crate_std"].isin(vuln_df["crate_std"])].copy()

# Label both sets
vuln_df["label"] = 1
safe_cleaned["label"] = 0

# Optional: Add a 'source' column for transparency
vuln_df["source"] = "vulnerable"
safe_cleaned["source"] = "safe_sample"

# Drop helper column
vuln_df = vuln_df.drop(columns=["crate_std"])
safe_cleaned = safe_cleaned.drop(columns=["crate_std"])

# Merge both
final_df = pd.concat([vuln_df, safe_cleaned], ignore_index=True)

# Save to CSV
final_output_path = "data/final_labeled_dataset.csv"
final_df.to_csv(final_output_path, index=False)

print(f"✅ Done! Final dataset created with {len(final_df)} crates.")
