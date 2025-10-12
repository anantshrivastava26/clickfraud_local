# eval_and_visualize.py
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURES_OUT = "session_features.csv"
MODEL_OUT = "isoforest_model.joblib"
SCALER_OUT = "scaler.joblib"

df = pd.read_csv(FEATURES_OUT)
print("Loaded", len(df), "sessions")
# load scaler & model
scaler = joblib.load(SCALER_OUT)
model = joblib.load(MODEL_OUT)

# select numeric features only
X = df.select_dtypes(include=[np.number]).fillna(0)

# ✅ Align columns with those seen by the scaler during training
if hasattr(scaler, 'feature_names_in_'):
    missing_cols = set(scaler.feature_names_in_) - set(X.columns)
    extra_cols = set(X.columns) - set(scaler.feature_names_in_)

    # add missing cols with 0s
    for c in missing_cols:
        X[c] = 0
    # drop unknown cols
    X = X[[c for c in scaler.feature_names_in_]]
else:
    print("⚠️ scaler missing feature_names_in_ (old sklearn version). Assuming same columns.")

X_scaled = scaler.transform(X)

# anomaly scores (the lower, the more anomalous for sklearn IsolationForest)
scores = model.decision_function(X_scaled)  # higher is normal, lower is outlier
pred = model.predict(X_scaled)  # -1 = outlier, 1 = inlier

df['anomaly_score'] = scores
df['is_outlier'] = (pred == -1).astype(int)

# basic counts
print(df['is_outlier'].value_counts())

# Plot histogram of anomaly scores
plt.figure(figsize=(8,4))
plt.hist(scores, bins=100)
plt.title("Anomaly Score Distribution")
plt.xlabel("IsolationForest decision_function score")
plt.ylabel("count")
plt.savefig("anomaly_score_hist.png")
print("Saved anomaly_score_hist.png")

# PCA 2D scatter
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(proj[:,0], proj[:,1], c=df['is_outlier'], cmap='coolwarm', s=6, alpha=0.6)
plt.title("PCA scatter - outliers highlighted (red)")
plt.savefig("pca_outliers.png")
print("Saved pca_outliers.png")

# Show top 50 anomalies for manual inspection
top_anom = df.sort_values('anomaly_score').head(50)
top_anom.to_csv("top_anomalies.csv", index=False)
print("Saved top_anomalies.csv")
