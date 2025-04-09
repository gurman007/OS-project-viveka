import pandas as pd

# Load safe crates (top 300/1000)
safe_df = pd.read_csv("data/top_crates_metadata.csv")
safe_df["vulnerable"] = 0

# Load risky crates (from RustSec + crates.io)
risky_df = pd.read_csv("data/rustsec_crates_metadata.csv")
risky_df["vulnerable"] = 1

# Remove duplicates if same crate appears in both
combined_df = pd.concat([safe_df, risky_df], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset=["crate"])

# Save final dataset
combined_df.to_csv("data/final_labeled_dataset.csv", index=False)
print(f"✅ Final dataset saved with {len(combined_df)} unique crates to data/final_balanced_dataset.csv")
