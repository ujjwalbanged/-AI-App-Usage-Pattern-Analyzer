"""Step 3: Data cleaning + basic EDA
Input : data/usage_data.csv
Output: data/usage_clean.csv + charts/*.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ---------- 1. LOAD ----------
df = pd.read_csv("usage_data.csv")
print("Raw shape:", df.shape)
print(df.dtypes, "\n")

# ---------- 2. CLEANING ----------
df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
print("Missing values:\n", df.isna().sum(), "\n")
df = df.dropna()
before = len(df)
df = df.drop_duplicates()
print("Duplicates removed:", before - len(df))

# invalid / outlier durations (0 se kam ya 3 ghante se zyada ek session)
df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 180)]

# ---------- 3. NEW TIME COLUMNS ----------
df["date"] = df["start_time"].dt.date
df["hour"] = df["start_time"].dt.hour
df["day_name"] = df["start_time"].dt.day_name()
df["is_weekend"] = df["start_time"].dt.dayofweek >= 5
df["is_night"] = (df["hour"] >= 22) | (df["hour"] < 5)

df.to_csv("usage_clean.csv", index=False)
print("Clean shape:", df.shape, "\n")

# ---------- 4. EDA NUMBERS ----------
daily = df.groupby(["user_id", "date"])["duration_min"].sum() / 60
print("Avg daily screen time (hrs):", round(daily.mean(), 2))
print("Night usage %:", round(df.loc[df.is_night, "duration_min"].sum() / df.duration_min.sum() * 100, 1))
print("\nTop 5 apps (total hrs):")
print((df.groupby("app_name")["duration_min"].sum() / 60).sort_values(ascending=False).head(5).round(0))

# ---------- 5. CHARTS ----------
# (a) Category share
cat = df.groupby("category")["duration_min"].sum()
plt.figure(figsize=(6, 6))
plt.pie(cat, labels=cat.index, autopct="%1.1f%%", startangle=90)
plt.title("Category-wise Usage Share")
plt.savefig("charts/1_category_share.png", dpi=120, bbox_inches="tight"); plt.close()

# (b) Top apps
apps = (df.groupby("app_name")["duration_min"].sum() / 60).sort_values()
plt.figure(figsize=(8, 5))
apps.plot(kind="barh", color="#4C72B0")
plt.xlabel("Total hours"); plt.title("Top Apps by Usage")
plt.savefig("charts/2_top_apps.png", dpi=120, bbox_inches="tight"); plt.close()

# (c) Hourly usage
hourly = df.groupby("hour")["duration_min"].sum() / 60
plt.figure(figsize=(9, 4))
hourly.plot(kind="bar", color="#DD8452")
plt.xlabel("Hour of day"); plt.ylabel("Total hours"); plt.title("Usage by Hour")
plt.savefig("charts/3_hourly_usage.png", dpi=120, bbox_inches="tight"); plt.close()

# (d) Heatmap: day x hour
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heat = df.pivot_table(index="day_name", columns="hour", values="duration_min", aggfunc="sum").reindex(order)
plt.figure(figsize=(11, 4))
sns.heatmap(heat, cmap="YlOrRd")
plt.title("Usage Heatmap (Day vs Hour)")
plt.savefig("charts/4_heatmap.png", dpi=120, bbox_inches="tight"); plt.close()

print("\nCharts saved in charts/")
