import pandas as pd
import requests
import time

# Load your dataset
df = pd.read_csv("data/final_labeled_dataset.csv")

# GitHub token goes here
GITHUB_TOKEN = "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"

# Extract repo slug from GitHub URL (owner/repo)
def extract_repo_slug(url):
    if pd.isna(url) or "github.com" not in url:
        return None
    url = url.replace(".git", "").strip("/")
    parts = url.split("github.com/")
    return parts[1] if len(parts) > 1 else None

# Headers for authenticated GitHub API
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Function to fetch GitHub repo metadata
def fetch_github_metadata(repo_slug):
    url = f"https://api.github.com/repos/{repo_slug}"
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            return {
                "repo_slug": repo_slug,
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
                "open_issues": data.get("open_issues_count", 0),
                "last_pushed": data.get("pushed_at")
            }
    except Exception as e:
        print(f"❌ Error fetching {repo_slug}: {e}")
    return {
        "repo_slug": repo_slug,
        "stars": None,
        "forks": None,
        "open_issues": None,
        "last_pushed": None
    }

# Apply on all rows with GitHub URLs
df["repo_slug"] = df["repository"].apply(extract_repo_slug)
slugs = df["repo_slug"].dropna().unique()

fetched = []
for slug in slugs:
    print(f"🔍 Fetching GitHub metadata for: {slug}")
    data = fetch_github_metadata(slug)
    print(f"✅ {slug} → ⭐ Stars: {data['stars']} | 🍴 Forks: {data['forks']} | 🐛 Issues: {data['open_issues']} | ⏳ Last Push: {data['last_pushed']}")
    fetched.append(data)
    time.sleep(1)  # Be nice to GitHub API


# Create DataFrame and merge
github_df = pd.DataFrame(fetched)
df = df.merge(github_df, on="repo_slug", how="left")

# Save final enriched file
df.to_csv("full_final_labeled_dataset.csv", index=False)
print("✅ Done! Saved as final_labeled_dataset_with_github.csv")
