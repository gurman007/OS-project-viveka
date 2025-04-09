import pandas as pd

# Load RustSec vulnerabilities
vuln_df = pd.read_csv("data/rustsec_vulnerabilities.csv")

# Drop duplicates and nulls, then sample 100 unique crates
unique_crates = vuln_df["crate"].dropna().drop_duplicates()
sampled = unique_crates.sample(n=100, random_state=42)

# Save to file
sampled.to_csv("data/selected_100_vulnerable_crates.csv", index=False)
print("✅ Saved 100 sampled vulnerable crates to data/selected_100_vulnerable_crates.csv")
