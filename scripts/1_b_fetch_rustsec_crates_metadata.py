import pandas as pd
import requests
import time

# 📥 Load RustSec vulnerability CSV
vuln_df = pd.read_csv("data/rustsec_vulnerabilities.csv")
unique_crates = vuln_df["crate"].dropna().str.strip().unique()

crate_data = []

print(f"📦 Fetching metadata for {len(unique_crates)} vulnerable crates from crates.io...")

for crate in unique_crates:
    url = f"https://crates.io/api/v1/crates/{crate}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()["crate"]
            crate_data.append({
                "crate": crate,
                "downloads": data.get("downloads", 0),
                "recent_downloads": data.get("recent_downloads", 0),
                "updated_at": data.get("updated_at"),
                "created_at": data.get("created_at"),
                "max_version": data.get("max_version"),
                "description": data.get("description", ""),
                "repository": data.get("repository", "")
            })
        else:
            print(f"❌ {crate}: Not found (status {response.status_code})")
    except Exception as e:
        print(f"⚠️ Error fetching {crate}: {e}")
    time.sleep(1)  # To be gentle with crates.io

# 💾 Save results
df = pd.DataFrame(crate_data)
df.to_csv("data/rustsec_crates_metadata.csv", index=False)
print("✅ Saved vulnerable crate metadata to: data/rustsec_crates_metadata.csv")
