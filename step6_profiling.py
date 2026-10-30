"""Step 6: Cluster profiling + auto naming + insights
Input : data/user_clusters.csv
Output: data/user_clusters_named.csv, data/cluster_profiles.csv,
        data/insights.txt, charts/7_radar.png, charts/8_cluster_bars.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("user_clusters.csv")

METRICS = ["avg_daily_hrs", "sessions_per_day", "night_pct", "weekend_diff_hrs",
           "social_pct", "games_pct", "productivity_pct", "entertainment_pct", "education_pct"]

# ---------- 1. PROFILE TABLE ----------
prof = df.groupby("cluster")[METRICS].mean().round(2)
prof["users"] = df.groupby("cluster").size()
prof["study_pct"] = prof["productivity_pct"] + prof["education_pct"]

# ---------- 2. AUTO NAMING (rule based, har naam sirf ek cluster ko) ----------
names, left = {}, list(prof.index)
rules = [("Night Owl", "night_pct"),          # sabse zyada raat ka usage
         ("Productive / Study", "study_pct"), # sabse zyada productivity+education
         ("Heavy Social", "avg_daily_hrs")]   # baaki mein sabse zyada screen time
for name, col in rules:
    c = prof.loc[left, col].idxmax()
    names[c] = name
    left.remove(c)
for c in left:
    names[c] = "Casual / Light"

prof.insert(0, "label", prof.index.map(names))
df["label"] = df["cluster"].map(names)
prof.to_csv("cluster_profiles.csv")
df.to_csv("user_clusters_named.csv", index=False)
print(prof[["label", "users", "avg_daily_hrs", "night_pct", "social_pct", "study_pct"]], "\n")

# ---------- 3. INSIGHTS (plain language) ----------
overall = df[METRICS].mean()
lines = []
for c, row in prof.iterrows():
    ins = [f"{row['label']} ({int(row['users'])} users):"]
    for col, txt in [("avg_daily_hrs", "daily screen time"), ("night_pct", "raat (10PM-5AM) ka usage"),
                     ("social_pct", "social apps ka share"), ("games_pct", "games ka share"),
                     ("productivity_pct", "productivity apps ka share")]:
        ratio = row[col] / overall[col]
        if ratio >= 1.5:
            ins.append(f"  - {txt} average se {ratio:.1f}x zyada hai ({row[col]:.1f} vs {overall[col]:.1f})")
        elif ratio <= 0.6:
            ins.append(f"  - {txt} average se kaafi kam hai ({row[col]:.1f} vs {overall[col]:.1f})")
    lines += ins + [""]
open("insights.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

# ---------- 4. RADAR CHART ----------
R = ["avg_daily_hrs", "night_pct", "social_pct", "games_pct", "productivity_pct", "education_pct"]
norm = (prof[R] - prof[R].min()) / (prof[R].max() - prof[R].min() + 1e-9)
ang = np.linspace(0, 2 * np.pi, len(R), endpoint=False).tolist(); ang += ang[:1]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for c, row in norm.iterrows():
    v = row.tolist() + row.tolist()[:1]
    ax.plot(ang, v, label=names[c], lw=2); ax.fill(ang, v, alpha=.1)
ax.set_xticks(ang[:-1]); ax.set_xticklabels([r.replace("_pct", " %").replace("_", " ") for r in R], fontsize=9)
ax.set_yticklabels([]); ax.set_title("Cluster Comparison (Radar)", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.savefig("charts/7_radar.png", dpi=120, bbox_inches="tight"); plt.close()

# ---------- 5. BAR CHART: category mix per cluster ----------
cats = ["social_pct", "entertainment_pct", "games_pct", "productivity_pct", "education_pct"]
mix = prof.set_index("label")[cats]
mix.columns = [c.replace("_pct", "").title() for c in cats]
mix.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="Set2")
plt.ylabel("% of usage time"); plt.xticks(rotation=0); plt.title("Category Mix by Cluster")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.savefig("charts/8_cluster_bars.png", dpi=120, bbox_inches="tight"); plt.close()
print("Charts saved.")
