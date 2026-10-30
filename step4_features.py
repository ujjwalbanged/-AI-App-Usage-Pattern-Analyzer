"""Step 4: Feature engineering (per-user features)
Input : data/usage_clean.csv
Output: data/user_features.csv  (1 row = 1 user)
"""
import pandas as pd

df = pd.read_csv("usage_clean.csv", parse_dates=["start_time"])
g = df.groupby("user_id")

# 1. Avg daily screen time (hours)
daily = df.groupby(["user_id", "date"])["duration_min"].sum() / 60
feat = pd.DataFrame({"avg_daily_hrs": daily.groupby("user_id").mean()})

# 2. Avg sessions per day
sessions = df.groupby(["user_id", "date"]).size()
feat["sessions_per_day"] = sessions.groupby("user_id").mean()

# 3. Avg session length (min)
feat["avg_session_min"] = g["duration_min"].mean()

# 4. Night usage % (10 PM - 5 AM)
total = g["duration_min"].sum()
night = df[df.is_night].groupby("user_id")["duration_min"].sum()
feat["night_pct"] = (night / total * 100).reindex(feat.index).fillna(0)

# 5. Peak usage hour (sabse zyada use wala ghanta)
hour_use = df.groupby(["user_id", "hour"])["duration_min"].sum().reset_index()
feat["peak_hour"] = hour_use.loc[hour_use.groupby("user_id")["duration_min"].idxmax()].set_index("user_id")["hour"]

# 6. Weekend vs weekday difference (hrs/day)
wk = df.groupby(["user_id", "is_weekend", "date"])["duration_min"].sum().div(60)
wk = wk.groupby(["user_id", "is_weekend"]).mean().unstack()
feat["weekend_diff_hrs"] = (wk[True] - wk[False])

# 7. Category share % (Social, Games, Productivity ...)
cat = df.pivot_table(index="user_id", columns="category", values="duration_min", aggfunc="sum").fillna(0)
cat = cat.div(cat.sum(axis=1), axis=0) * 100
cat.columns = [f"{c.lower()}_pct" for c in cat.columns]
feat = feat.join(cat)

feat = feat.round(2).reset_index()
feat.to_csv("user_features.csv", index=False)

print(feat.shape)
print(feat.head())
print("\nSummary:")
print(feat.drop(columns="user_id").describe().loc[["mean", "min", "max"]].round(2).T)
