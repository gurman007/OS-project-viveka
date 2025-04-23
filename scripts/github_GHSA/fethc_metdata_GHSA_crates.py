import pandas as pd
import requests
import time

# Load your GHSA crates (with missing metadata)
df = pd.read_csv("combined_vulnerable_augmented.csv")
ghsa_df = df[df["downloads"].isna()]

# Function to fetch crate metadata
def fetch_crate_metadata(crate_name):
    url = f"https://crates.io/api/v1/crates/{crate_name}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()["crate"]
            return {
                "crate": crate_name,
                "downloads": data["downloads"],
                "recent_downloads": data.get("recent_downloads", 0),
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "max_version": data["max_version"],
                "description": data.get("description", ""),
                "repository": data.get("repository", "")
            }
    except Exception as e:
        print(f"❌ Failed: {crate_name}, error: {e}")
    return {key: None for key in ["crate", "downloads", "recent_downloads", "created_at", "updated_at", "max_version", "description", "repository"]}

# Fetch all
results = []
for crate in ghsa_df["crate"].dropna().unique():
    print(f"🔍 Fetching: {crate}")
    metadata = fetch_crate_metadata(crate)
    metadata["crate"] = crate
    results.append(metadata)
    time.sleep(1)  # Be kind to crates.io

# Save result
pd.DataFrame(results).to_csv("ghsa_crates_fetched_metadata.csv", index=False)
print("✅ Done! Saved to ghsa_crates_fetched_metadata.csv")
