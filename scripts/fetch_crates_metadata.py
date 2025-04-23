import requests  # <-- This line is essential.
import pandas as pd
import time

# Configuration
TOTAL_CRATES = 1500
PER_PAGE = 100  # Maximum allowed per page by crates.io

all_crates = []

print("📦 Fetching top crates from crates.io...")

for page in range(1, (TOTAL_CRATES // PER_PAGE) + 1):
    url = f"https://crates.io/api/v1/crates?page={page}&per_page={PER_PAGE}&sort=downloads"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to fetch page {page} (Status code: {response.status_code})")
        continue
    
    crates = response.json()["crates"]
    for crate in crates:
        all_crates.append({
            "crate": crate["id"],
            "downloads": crate["downloads"],
            "recent_downloads": crate.get("recent_downloads", 0),
            "updated_at": crate["updated_at"],
            "created_at": crate["created_at"],
            "max_version": crate["max_version"],
            "description": crate.get("description", ""),
            "repository": crate.get("repository", "")  # <-- New field for GitHub URL
        })
    
    time.sleep(1)  # Pause briefly to be kind to the API

# Save to CSV in the data folder
df = pd.DataFrame(all_crates)
df.to_csv("data/top_crates_metadata.csv", index=False)
print("✅ Updated metadata with GitHub repositories saved to data/top_crates_metadata.csv")
