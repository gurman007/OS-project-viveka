import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ Load your full labeled dataset
df = pd.read_csv("full_final_labeled_dataset.csv")

# 2️⃣ Split: 95% train, 5% test
train_df, test_df = train_test_split(df, test_size=0.05, random_state=42, shuffle=True)

# 3️⃣ Save to separate files
train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)

print(f"✅ Training set saved: {len(train_df)} rows → data/train_data.csv")
print(f"✅ Test set saved: {len(test_df)} rows → data/test_data.csv")
