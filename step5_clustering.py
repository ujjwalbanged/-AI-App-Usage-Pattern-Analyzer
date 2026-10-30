"""Step 5: K-Means clustering
Input : data/user_features.csv
Output: data/user_clusters.csv + charts/5_elbow_silhouette.png + charts/6_clusters_pca.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv("user_features.csv")

# ---------- 1. FEATURE SELECTION ----------
# avg_session_min hataya (sabke liye almost same)
# peak_hour ko sin/cos mein badla (circular hai: 23 aur 0 paas hain)
df["peak_sin"] = np.sin(2 * np.pi * df["peak_hour"] / 24)
df["peak_cos"] = np.cos(2 * np.pi * df["peak_hour"] / 24)

FEATURES = ["avg_daily_hrs", "sessions_per_day", "night_pct", "weekend_diff_hrs",
            "social_pct", "games_pct", "productivity_pct", "entertainment_pct",
            "education_pct", "peak_sin", "peak_cos"]

# ---------- 2. SCALING ----------
X = StandardScaler().fit_transform(df[FEATURES])

# ---------- 3. BEST K (Elbow + Silhouette) ----------
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    print(f"K={k}  inertia={km.inertia_:.0f}  silhouette={sil[-1]:.3f}")

best_k = list(ks)[int(np.argmax(sil))]
print("\nBest K (max silhouette):", best_k)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow Method"); ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil, "o-", color="green"); ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("K")
ax[1].axvline(best_k, ls="--", color="red")
plt.tight_layout(); plt.savefig("charts/5_elbow_silhouette.png", dpi=120); plt.close()

# ---------- 4. FINAL MODEL ----------
K = 4  # silhouette K=4 (0.593) aur K=5 (0.597) lagbhag same; 4 interpret karna aasan hai
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
df["cluster"] = km.labels_
print("\nCluster sizes:\n", df["cluster"].value_counts().sort_index())

# ---------- 5. PCA 2D PLOT ----------
pca = PCA(n_components=2)
pts = pca.fit_transform(X)
plt.figure(figsize=(7, 6))
for c in range(K):
    m = df["cluster"] == c
    plt.scatter(pts[m, 0], pts[m, 1], label=f"Cluster {c}", s=40, alpha=.8)
plt.legend(); plt.title(f"User Clusters (PCA, K={K})")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
plt.savefig("charts/6_clusters_pca.png", dpi=120, bbox_inches="tight"); plt.close()

df.to_csv("user_clusters.csv", index=False)

# ---------- 6. QUICK PROFILE ----------
print("\nCluster averages:")
print(df.groupby("cluster")[["avg_daily_hrs", "night_pct", "social_pct", "games_pct",
                              "productivity_pct", "education_pct"]].mean().round(1))
