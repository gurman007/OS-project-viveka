import os
import toml
import pandas as pd

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "advisory-db", "crates"))
data = []

print("🔍 Walking through:", base_path)

for crate in os.listdir(base_path):
    crate_dir = os.path.join(base_path, crate)
    if os.path.isdir(crate_dir):
        for file in os.listdir(crate_dir):
            if file.endswith(".md"):
                path = os.path.join(crate_dir, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extract TOML front matter
                        toml_start = content.find('```toml')
                        toml_end = content.find('```', toml_start + 7)
                        if toml_start != -1 and toml_end != -1:
                            toml_content = content[toml_start + 7:toml_end].strip()
                            advisory_data = toml.loads(toml_content)
                            if "advisory" in advisory_data:
                                a = advisory_data["advisory"]
                                data.append({
                                    "crate": a.get("package"),
                                    "id": a.get("id"),
                                    "title": a.get("title"),
                                    "description": a.get("title"),  # Adjust as needed
                                    "date": a.get("date")
                                })
                            else:
                                print(f"⚠️ No [advisory] section in {file}")
                        else:
                            print(f"⚠️ No TOML front matter found in {file}")
                except Exception as e:
                    print(f"❌ Failed to parse {file}: {e}")

print(f"✅ Parsed {len(data)} advisories")
df = pd.DataFrame(data)
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "rustsec_vulnerabilities.csv"))
df.to_csv(output_path, index=False)
print("✅ Saved CSV to:", output_path)
