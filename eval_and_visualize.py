# eval_and_visualize.py
"""
Offline sanity check of the trained model over the full training-session table.

Reads  : session_features.csv, scaler.joblib, isoforest_model.joblib  (from training)
Writes : anomaly_score_hist.png   histogram of decision_function scores
         pca_outliers.png         2-D PCA scatter, outliers in red
         top_anomalies.csv        the 50 lowest-scoring (most anomalous) sessions

There are no fraud labels, so this reports how many sessions the model flags and how the
scores are distributed - not precision/recall. Eyeball top_anomalies.csv to judge whether
the flagged sessions look like genuine extremes.

This is a plain top-level script (no functions / __main__ guard): run it directly with
`python eval_and_visualize.py` from the repo root.
"""
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

FEATURES_OUT = "session_features.csv"
MODEL_OUT = "isoforest_model.joblib"
SCALER_OUT = "scaler.joblib"

# Load features
df = pd.read_csv(FEATURES_OUT)
print("Loaded", len(df), "sessions")

# ✅ Load scaler & model (handle both old/new formats)
# New format: dict with the scaler and the ordered list of columns it was fitted on.
# Old format: a bare StandardScaler, so fall back to whatever the scaler/CSV offers.
scaler_blob = joblib.load(SCALER_OUT)
if isinstance(scaler_blob, dict):
    scaler = scaler_blob["scaler"]
    feature_names = scaler_blob["features"]
else:
    scaler = scaler_blob
    feature_names = getattr(scaler, "feature_names_in_", list(df.select_dtypes(include=[np.number]).columns))

model = joblib.load(MODEL_OUT)

# Select numeric features
X = df.select_dtypes(include=[np.number]).fillna(0)

# ✅ Align columns with training features
# The CSV holds every computed feature, but the model only saw the non-constant ones.
# Selecting those, in training order, is mandatory: scaler.transform is positional.
X = X.reindex(columns=feature_names, fill_value=0)

# Scale
X_scaled = scaler.transform(X)

# Predict anomalies
scores = model.decision_function(X_scaled)  # higher = normal, lower = anomaly
pred = model.predict(X_scaled)  # -1 = anomaly, 1 = normal

# Add results to DataFrame
df['anomaly_score'] = scores
df['is_outlier'] = (pred == -1).astype(int)

# Print basic stats
# Because training used contamination=0.02 and we score the SAME data, this rate should
# land near 2% - it reflects that setting, not a measured fraud rate.
print(df['is_outlier'].value_counts())
print(f"Suspicious rate: {(df['is_outlier'].mean() * 100):.2f}%")

# 📊 Plot histogram of anomaly scores
# Scores below 0 are outliers; a long left tail = a few very extreme sessions.
plt.figure(figsize=(8,4))
plt.hist(scores, bins=100, color='steelblue', alpha=0.8)
plt.title("Anomaly Score Distribution")
plt.xlabel("IsolationForest decision_function score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("anomaly_score_hist.png")
print("Saved anomaly_score_hist.png")

# 🧭 PCA visualization
# Projects the scaled feature space to 2-D purely for plotting (the model does NOT use
# PCA). Two components only capture part of the variance, so overlap is expected.
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(proj[:,0], proj[:,1], c=df['is_outlier'], cmap='coolwarm', s=8, alpha=0.7)
plt.title("PCA Projection — Outliers in Red")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_outliers.png")
print("Saved pca_outliers.png")

# Save top anomalies
# Ascending sort: the lowest score is the most anomalous.
top_anom = df.sort_values('anomaly_score').head(50)
top_anom.to_csv("top_anomalies.csv", index=False)
print("Saved top_anomalies.csv")
