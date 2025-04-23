import pandas as pd

# Load both files
main_df = pd.read_csv("combined_vulnerable_augmented.csv")
ghsa_fetched_df = pd.read_csv("ghsa_crates_fetched_metadata.csv")

# Merge based on crate name (update only missing values)
merged_df = main_df.merge(
    ghsa_fetched_df,
    on="crate",
    how="left",
    suffixes=("", "_ghsa")
)

# Fill missing fields from GHSA data
for col in ["downloads", "recent_downloads", "created_at", "updated_at", "max_version", "description", "repository"]:
    merged_df[col] = merged_df[col].combine_first(merged_df[f"{col}_ghsa"])

# Drop temp columns
merged_df = merged_df[[col for col in merged_df.columns if not col.endswith("_ghsa")]]

# Save final merged file
merged_df.to_csv("fully_enriched_vulnerable_dataset.csv", index=False)
print("✅ Merged GHSA data into main dataset! Saved to fully_enriched_vulnerable_dataset.csv")
