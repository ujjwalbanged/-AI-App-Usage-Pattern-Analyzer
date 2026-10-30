"""Step 2: Synthetic app usage data generator.
Output: data/usage_data.csv
Columns: user_id, app_name, category, start_time, duration_min
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

APPS = {
    "Instagram": "Social", "WhatsApp": "Social", "YouTube": "Entertainment",
    "Netflix": "Entertainment", "PUBG": "Games", "Chrome": "Productivity",
    "Google Docs": "Productivity", "Duolingo": "Education", "Coursera": "Education",
}
CATS = {}
for a, c in APPS.items():
    CATS.setdefault(c, []).append(a)

# Har archetype: (sessions/day, peak hours, category weights)
PROFILES = {
    "heavy_social": dict(n=(25, 40), hours=[13, 14, 18, 19, 20, 21],
                         w={"Social": .55, "Entertainment": .25, "Games": .1, "Productivity": .05, "Education": .05}),
    "productive":   dict(n=(12, 20), hours=[9, 10, 11, 14, 15, 16],
                         w={"Social": .1, "Entertainment": .1, "Games": .03, "Productivity": .45, "Education": .32}),
    "night_owl":    dict(n=(18, 30), hours=[22, 23, 0, 1, 2, 3],
                         w={"Social": .35, "Entertainment": .35, "Games": .2, "Productivity": .05, "Education": .05}),
    "casual":       dict(n=(5, 10), hours=[8, 13, 19, 20],
                         w={"Social": .35, "Entertainment": .2, "Games": .1, "Productivity": .2, "Education": .15}),
}

def make_user(uid, kind, days=30, start="2025-01-01"):
    p = PROFILES[kind]
    cats, probs = list(p["w"]), np.array(list(p["w"].values()))
    rows = []
    for d in pd.date_range(start, periods=days):
        weekend = d.dayofweek >= 5
        n = rng.integers(*p["n"]) + (5 if weekend else 0)
        for _ in range(n):
            cat = rng.choice(cats, p=probs / probs.sum())
            app = rng.choice(CATS[cat])
            hour = int(rng.choice(p["hours"]) if rng.random() < .75 else rng.integers(0, 24))
            t = d + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            dur = max(1, round(rng.gamma(2.0, 6.0) * (1.3 if weekend else 1), 1))
            rows.append((uid, app, cat, t, dur))
    return rows

rows = []
uid = 1
for kind in PROFILES:
    for _ in range(50):  # 50 users per type = 200 users
        rows += make_user(f"U{uid:03d}", kind)
        uid += 1

df = pd.DataFrame(rows, columns=["user_id", "app_name", "category", "start_time", "duration_min"])
df = df.sample(frac=1, random_state=1).sort_values(["user_id", "start_time"])
df.to_csv("usage_data.csv", index=False)
print(df.shape, "users:", df.user_id.nunique())
print(df.head())
