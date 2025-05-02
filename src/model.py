# train_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Simulated dataset
data = {
    "origin": ["JFK", "LAX", "ORD", "JFK", "LAX", "ORD", "ATL", "ATL", "LAX"],
    "destination": ["ATL", "JFK", "LAX", "ORD", "ORD", "JFK", "LAX", "JFK", "ORD"],
    "dep_hour": [8, 14, 20, 6, 18, 21, 9, 10, 15],
    "delayed": [0, 1, 1, 0, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(data)

# Encode categories manually
origin_map = {val: idx for idx, val in enumerate(df["origin"].unique())}
dest_map = {val: idx for idx, val in enumerate(df["destination"].unique())}

df["origin_enc"] = df["origin"].map(origin_map)
df["dest_enc"] = df["destination"].map(dest_map)

# Train model
X = df[["origin_enc", "dest_enc", "dep_hour"]]
y = df["delayed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and options
joblib.dump(model, "model.pkl")
joblib.dump({
    "origin_map": origin_map,
    "dest_map": dest_map,
    "origin_options": list(origin_map.keys()),
    "dest_options": list(dest_map.keys()),
}, "options.pkl")