import pandas as pd
import requests
import time
from datetime import datetime, timezone

# 🔐 STEP 1: Add your GitHub Token here (no quotes missing!)
GITHUB_TOKEN = "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"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}


# 📂 STEP 2: Load top crates metadata
metadata_filepath = "data/top_crates_metadata.csv"
df = pd.read_csv(metadata_filepath)

# 🧹 STEP 3: Clean and filter GitHub repository links
df["repository"] = df["repository"].fillna("").str.strip()
df["repository"] = df["repository"].str.replace(".git", "", regex=False).str.rstrip("/")
df = df[df["repository"].str.startswith("https://github.com/")].copy()
print(f"🌐 Found {len(df)} crates with valid GitHub repositories")

# 🧱 STEP 4: Initialize new GitHub feature columns
df["stars"] = None
df["forks"] = None
df["open_issues"] = None
df["license"] = None

# 🔁 STEP 5: GitHub API fetch function with retry logic
def fetch_github_info(repo_url, max_retries=3):
    owner_repo = repo_url.replace("https://github.com/", "").strip("/")
    api_url = f"https://api.github.com/repos/{owner_repo}"
    
    for attempt in range(max_retries):
        try:
            res = requests.get(api_url, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                data = res.json()
                lic = data.get("license")
                license_name = lic["name"] if lic and "name" in lic else None
                return {
                    "stars": data.get("stargazers_count", 0),
                    "forks": data.get("forks_count", 0),
                    "open_issues": data.get("open_issues_count", 0),
                    "license": license_name
                }
            elif res.status_code == 404:
                print(f"❌ Not found: {repo_url}")
                return None
            else:
                print(f"⚠️ API error {res.status_code} for {repo_url} → retrying ({attempt + 1})")
        except Exception as e:
            print(f"❌ Error fetching {repo_url} (attempt {attempt + 1}): {e}")
        time.sleep(2)
    return None

# 🚀 STEP 6: Enrich crate data
print("🔍 Enriching GitHub data...")
for idx, row in df.iterrows():
    repo_url = row["repository"]
    info = fetch_github_info(repo_url)
    if info:
        df.at[idx, "stars"] = info["stars"]
        df.at[idx, "forks"] = info["forks"]
        df.at[idx, "open_issues"] = info["open_issues"]
        df.at[idx, "license"] = info["license"]
        print(f"✅ {repo_url} → ⭐ {info['stars']} | 🍴 {info['forks']} | 🐞 {info['open_issues']} | 🔖 {info['license']}")
    else:
        print(f"⚠️ Skipped: {repo_url}")
    time.sleep(1.5)

# 📊 STEP 7: Compute additional features
df["active_ratio"] = df["recent_downloads"] / df["downloads"].replace(0, 1)

df["updated_at_dt"] = pd.to_datetime(df["updated_at"], errors='coerce', utc=True)
df["days_since_update"] = (datetime.now(timezone.utc) - df["updated_at_dt"]).dt.days

df["created_at_dt"] = pd.to_datetime(df["created_at"], errors='coerce', utc=True)
df["days_since_created"] = (datetime.now(timezone.utc) - df["created_at_dt"]).dt.days

# ➕ STEP 8: Flag rows where GitHub data is missing
df["github_data_missing"] = df["stars"].isna().astype(int)

# 💾 STEP 9: Save the enriched dataset
out_file = "data/top_crates_with_github_enriched.csv"
df.to_csv(out_file, index=False)
print(f"\n✅ Enriched dataset saved to: {out_file}")
