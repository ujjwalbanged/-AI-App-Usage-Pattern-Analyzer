"""Step 7: Streamlit dashboard  ->  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI App Usage Pattern Analyzer", page_icon="📱", layout="wide")

FEATURES = ["avg_daily_hrs", "sessions_per_day", "night_pct", "weekend_diff_hrs",
            "social_pct", "games_pct", "productivity_pct", "entertainment_pct",
            "education_pct", "peak_sin", "peak_cos"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_data
def load():
    usage = pd.read_csv("usage_clean.csv", parse_dates=["start_time"])
    users = pd.read_csv("user_clusters_named.csv")
    prof = pd.read_csv("cluster_profiles.csv")
    users = users.merge(usage.groupby("user_id")["app_name"].agg(lambda s: s.value_counts().index[0])
                        .rename("top_app"), on="user_id")
    X = StandardScaler().fit_transform(users[FEATURES])
    pts = PCA(n_components=2).fit_transform(X)
    users["PC1"], users["PC2"] = pts[:, 0], pts[:, 1]
    return usage, users, prof


try:
    usage, users, prof = load()
except FileNotFoundError:
    st.error("Data files nahi mili. Pehle step3 se step6 tak ki scripts run karo.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📱 Usage Analyzer")
page = st.sidebar.radio("Page", ["Overview", "Usage Patterns", "Clusters", "My Type"])
labels = sorted(users["label"].unique())
sel = st.sidebar.multiselect("Cluster filter", labels, default=labels)
ids = users[users["label"].isin(sel)]["user_id"]
u = usage[usage["user_id"].isin(ids)]
us = users[users["user_id"].isin(ids)]
if u.empty:
    st.warning("Kam se kam ek cluster chuno.")
    st.stop()

# ---------------- PAGE 1: OVERVIEW ----------------
if page == "Overview":
    st.title("Overview")
    daily = u.groupby(["user_id", "date"])["duration_min"].sum() / 60
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{u.user_id.nunique()}")
    c2.metric("Avg daily screen time", f"{daily.mean():.2f} hrs")
    c3.metric("Night usage", f"{u.loc[u.is_night, 'duration_min'].sum() / u.duration_min.sum() * 100:.1f}%")
    c4.metric("Total sessions", f"{len(u):,}")

    a, b = st.columns(2)
    cat = u.groupby("category")["duration_min"].sum().reset_index()
    a.plotly_chart(px.pie(cat, names="category", values="duration_min", title="Category share", hole=.4),
                   width="stretch")
    apps = (u.groupby("app_name")["duration_min"].sum() / 60).sort_values().reset_index()
    b.plotly_chart(px.bar(apps, x="duration_min", y="app_name", orientation="h",
                          title="Top apps (total hours)", labels={"duration_min": "Hours", "app_name": ""}),
                   width="stretch")

# ---------------- PAGE 2: USAGE PATTERNS ----------------
elif page == "Usage Patterns":
    st.title("Usage Patterns")
    hourly = (u.groupby("hour")["duration_min"].sum() / 60).reset_index()
    st.plotly_chart(px.bar(hourly, x="hour", y="duration_min", title="Usage by hour of day",
                           labels={"duration_min": "Hours"}), width="stretch")

    heat = u.pivot_table(index="day_name", columns="hour", values="duration_min", aggfunc="sum").reindex(DAY_ORDER)
    st.plotly_chart(px.imshow(heat, color_continuous_scale="YlOrRd", aspect="auto",
                              title="Heatmap: day vs hour", labels=dict(color="Minutes")),
                    width="stretch")

    wk = u.groupby(["user_id", "is_weekend", "date"])["duration_min"].sum().div(60) \
          .groupby(["user_id", "is_weekend"]).mean().groupby("is_weekend").mean().rename(
              {False: "Weekday", True: "Weekend"}).reset_index()
    wk.columns = ["Type", "Avg hrs/day"]
    st.plotly_chart(px.bar(wk, x="Type", y="Avg hrs/day", title="Weekday vs weekend", color="Type"),
                    width="stretch")

# ---------------- PAGE 3: CLUSTERS ----------------
elif page == "Clusters":
    st.title("Clusters")
    a, b = st.columns(2)
    a.plotly_chart(px.scatter(us, x="PC1", y="PC2", color="label", hover_data=["user_id", "avg_daily_hrs"],
                              title="User clusters (PCA 2D)"), width="stretch")
    R = ["avg_daily_hrs", "night_pct", "social_pct", "games_pct", "productivity_pct", "education_pct"]
    p = prof[prof["label"].isin(sel)].set_index("label")[R]
    norm = (p - p.min()) / (p.max() - p.min() + 1e-9)
    fig = go.Figure()
    for lab, row in norm.iterrows():
        fig.add_trace(go.Scatterpolar(r=row.tolist() + [row.iloc[0]], theta=R + [R[0]], name=lab, fill="toself", opacity=.5))
    fig.update_layout(title="Cluster comparison (radar)")
    b.plotly_chart(fig, width="stretch")

    st.subheader("Cluster profiles")
    show = prof[prof["label"].isin(sel)].drop(columns=["cluster"], errors="ignore").set_index("label").round(1)
    st.dataframe(show, width="stretch")
    st.subheader("Insights")
    try:
        st.text(open("data/insights.txt", encoding="utf-8").read())
    except FileNotFoundError:
        st.info("insights.txt nahi mili (step6 run karo).")

# ---------------- PAGE 4: MY TYPE ----------------
else:
    st.title("My Type")
    uid = st.selectbox("User ID chuno", users["user_id"])
    me = users[users["user_id"] == uid].iloc[0]
    st.success(f"**{uid} → {me['label']}**   |   Top app: {me['top_app']}")

    c1, c2, c3 = st.columns(3)
    overall = users[FEATURES].mean()
    c1.metric("Daily screen time", f"{me.avg_daily_hrs:.1f} hrs", f"{me.avg_daily_hrs - overall.avg_daily_hrs:+.1f} vs avg", delta_color="off")
    c2.metric("Night usage", f"{me.night_pct:.0f}%", f"{me.night_pct - overall.night_pct:+.0f}% vs avg", delta_color="off")
    c3.metric("Social share", f"{me.social_pct:.0f}%", f"{me.social_pct - overall.social_pct:+.0f}% vs avg", delta_color="off")

    M = ["avg_daily_hrs", "night_pct", "social_pct", "games_pct", "productivity_pct", "education_pct"]
    cl = users[users["label"] == me["label"]][M].mean()
    cmp = pd.DataFrame({"You": me[M].astype(float), "Your cluster avg": cl, "All users avg": overall[M]}).reset_index()
    cmp = cmp.melt(id_vars="index", var_name="Who", value_name="Value")
    st.plotly_chart(px.bar(cmp, x="index", y="Value", color="Who", barmode="group",
                           title="You vs your group vs everyone", labels={"index": ""}), width="stretch")

    st.subheader("Your insights")
    tips = []
    for col, txt in [("avg_daily_hrs", "daily screen time"), ("night_pct", "raat ka usage"),
                     ("social_pct", "social apps ka share"), ("games_pct", "games ka share"),
                     ("productivity_pct", "productivity apps ka share")]:
        r = me[col] / overall[col]
        if r >= 1.5:
            tips.append(f"⬆️ Aapka {txt} average se **{r:.1f}x** zyada hai.")
        elif r <= 0.6:
            tips.append(f"⬇️ Aapka {txt} average se kaafi kam hai.")
    st.write("\n\n".join(tips) if tips else "Aapka usage average ke kaafi paas hai.")
