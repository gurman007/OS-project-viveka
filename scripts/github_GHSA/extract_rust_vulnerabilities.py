import os
import json
import pandas as pd

base_path = "advisory-database/advisories/github-reviewed"
vulnerable_data = []
total_files = 0
rust_entries = 0

print("🔍 Starting scan for Rust crate vulnerabilities...\n")

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".json"):
            total_files += 1
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    advisory = json.load(f)

                    advisory_id = advisory.get("id")
                    description = advisory.get("description", "")
                    date = advisory.get("published", "")
                    cvss_score = advisory.get("cvss", {}).get("score", None)

                    affected_packages = advisory.get("affected", [])
                    for pkg in affected_packages:
                        ecosystem = pkg.get("package", {}).get("ecosystem")
                        crate_name = pkg.get("package", {}).get("name")

                        if ecosystem == "crates.io" and crate_name:
                            rust_entries += 1
                            print(f"✅ Found vulnerable crate: {crate_name}  ({advisory_id})")
                            vulnerable_data.append({
                                "crate_name": crate_name,
                                "advisory_id": advisory_id,
                                "description": description,
                                "published_date": date,
                                "cvss_score": cvss_score,
                                "label": 1
                            })
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")

# Save to CSV
df = pd.DataFrame(vulnerable_data)
df.to_csv("GHSA_vulnerable_crates.csv", index=False)

print("\n✅ Scan Complete!")
print(f"📂 Total .json files scanned: {total_files}")
print(f"🐍 Rust crate vulnerabilities found: {rust_entries}")
print(f"📄 Records saved to vulnerable_crates.csv: {len(df)}")
