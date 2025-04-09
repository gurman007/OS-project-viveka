import os

print("📂 Current working directory:", os.getcwd())

if os.path.exists("data/final_labeled_dataset.csv"):
    print("✅ File found.")
else:
    print("❌ File NOT found in 'data/' folder.")
