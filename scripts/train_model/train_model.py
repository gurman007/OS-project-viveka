import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

import joblib

# 1️⃣ Load your labeled dataset
df = pd.read_csv("data/final_labeled_dataset.csv")

# 2️⃣ Define the features to use
features = [
    "downloads", "recent_downloads", "stars", "forks", "open_issues",
    "active_ratio", "days_since_update", "days_since_created"
]

X = df[features]
y = df["vulnerable"]

# 3️⃣ Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Combine X_train and y_train to prepare for upsampling
train_df = X_train.copy()
train_df["vulnerable"] = y_train

# 5️⃣ Separate safe and risky crates
safe = train_df[train_df["vulnerable"] == 0]
risky = train_df[train_df["vulnerable"] == 1]

# 6️⃣ Upsample risky crates to match number of safe ones
risky_upsampled = resample(
    risky,
    replace=True,
    n_samples=len(safe),
    random_state=42
)

# 7️⃣ Combine back the balanced dataset
upsampled_df = pd.concat([safe, risky_upsampled])
X_train_bal = upsampled_df[features]
y_train_bal = upsampled_df["vulnerable"]

print(f"✅ After upsampling: {len(X_train_bal)} total training samples (safe: {len(safe)}, risky: {len(risky_upsampled)})")

# 8️⃣ Train Random Forest on balanced data
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train_bal, y_train_bal)

# 9️⃣ Evaluate on the original test set
y_pred = model.predict(X_test)

print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

