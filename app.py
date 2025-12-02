import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Traffic Lens AI", page_icon="🚦", layout="wide")

PRIMARY = "#FF4B4B"
SECONDARY = "#1E1E1E"
ACCENT = "#FFD700"
BG_COLOR = "#F5F5F5"
TEXT_COLOR = "#222222"

# Smooth fade-in appearance
st.markdown("""
<style>
@keyframes fadeSmooth {
    from { opacity:0; transform:translateY(10px); }
    to { opacity:1; transform:translateY(0px); }
}
div { animation: fadeSmooth 0.4s ease-in-out; }
body { background-color: #F5F5F5; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_default():
    df = pd.read_csv("simulated_traffic_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
else:
    df = load_default()

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.subheader("Filters")

min_date = df["Timestamp"].min().date()
max_date = df["Timestamp"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

start_date, end_date = date_range
df = df[(df["Timestamp"].dt.date >= start_date) & (df["Timestamp"].dt.date <= end_date)]

states = sorted(df["State"].unique())
selected_state = st.sidebar.selectbox("State", states)

cities = sorted(df[df["State"] == selected_state]["City"].unique())
selected_city = st.sidebar.selectbox("City", ["All Cities"] + cities)

df_filtered = df[df["State"] == selected_state]
if selected_city != "All Cities":
    df_filtered = df_filtered[df_filtered["City"] == selected_city]

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🗺 Maps",
    "🤖 ML Predictions",
    "📁 Upload Data"
])

# =========================================================
# TAB 1 — DASHBOARD
# =========================================================
with tab1:

    st.header(f"📊 Traffic Overview – {selected_state}")

    avg_traffic = df_filtered["VehicleCount"].mean()
    peak_hour = df_filtered.groupby("HourOfDay")["VehicleCount"].mean().idxmax()
    max_traffic = df_filtered["VehicleCount"].max()

    # FIXED KPI BOXES (HTML rendering)
    kpi_html = f"""
    <div style="display:flex; gap:20px; justify-content:center; margin:20px 0;">

        <div style="flex:1; background:white; padding:20px; 
            border-left:8px solid {PRIMARY}; border-radius:10px; 
            box-shadow:0 4px 10px rgba(0,0,0,0.1);">
            <h3 style="margin:0; color:{PRIMARY};">Average Traffic</h3>
            <p style="font-size:28px; font-weight:800; margin:5px 0;">{avg_traffic:.1f}</p>
        </div>

        <div style="flex:1; background:white; padding:20px; 
            border-left:8px solid {ACCENT}; border-radius:10px;
            box-shadow:0 4px 10px rgba(0,0,0,0.1);">
            <h3 style="margin:0; color:{ACCENT};">Peak Hour</h3>
            <p style="font-size:28px; font-weight:800; margin:5px 0;">{peak_hour}:00</p>
        </div>

        <div style="flex:1; background:white; padding:20px; 
            border-left:8px solid {SECONDARY}; border-radius:10px;
            box-shadow:0 4px 10px rgba(0,0,0,0.1);">
            <h3 style="margin:0; color:{SECONDARY};">Max Vehicles</h3>
            <p style="font-size:28px; font-weight:800; margin:5px 0;">{max_traffic}</p>
        </div>

    </div>
    """

    st.markdown(kpi_html, unsafe_allow_html=True)

    # Line Graph
    st.subheader("📈 Traffic Over Time")
    fig1 = px.line(df_filtered, x="Timestamp", y="VehicleCount", color="City")
    st.plotly_chart(fig1, use_container_width=True)

    # Heatmap
    st.subheader("🔥 Traffic Heatmap")
    heatmap_df = df_filtered.groupby(["DayOfWeek", "HourOfDay"])["VehicleCount"].mean().reset_index()
    fig_heat = px.density_heatmap(
        heatmap_df, x="HourOfDay", y="DayOfWeek",
        z="VehicleCount", color_continuous_scale="Inferno"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Compare Cities
    st.subheader("🏙 Compare Two Cities")
    c1 = st.selectbox("City 1", cities)
    c2 = st.selectbox("City 2", cities, index=1)
    compare_df = df[df["City"].isin([c1, c2])]
    fig_cmp = px.line(compare_df, x="Timestamp", y="VehicleCount", color="City")
    st.plotly_chart(fig_cmp, use_container_width=True)

# =========================================================
# TAB 2 — MAPS
# =========================================================
with tab2:

    st.header("🗺 Traffic Maps")

    city_coords = {
        "Hartford": (41.7658, -72.6734), "New Haven": (41.3083, -72.9279),
        "Stamford": (41.0534, -73.5387), "Boston": (42.3601, -71.0589),
        "Worcester": (42.2626, -71.8023), "Springfield": (42.1015, -72.5898),
        "Newark": (40.7357, -74.1724), "Jersey City": (40.7178, -74.0431),
        "Paterson": (40.9168, -74.1718), "New York City": (40.7128, -74.0060),
        "Buffalo": (42.8864, -78.8784), "Rochester": (43.1566, -77.6088)
    }

    df_map = df.copy()
    df_map["Lat"] = df_map["City"].map(lambda c: city_coords[c][0])
    df_map["Lon"] = df_map["City"].map(lambda c: city_coords[c][1])

    st.subheader("🗽 NYC Traffic Map")
    fig_nyc = px.scatter_mapbox(
        df_map, lat="Lat", lon="Lon",
        size="VehicleCount", color="VehicleCount",
        zoom=6, color_continuous_scale="Turbo"
    )
    fig_nyc.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_nyc, use_container_width=True)

    st.subheader("🎞 Animated Traffic Map")
    fig_anim = px.scatter_mapbox(
        df_map, lat="Lat", lon="Lon", animation_frame="HourOfDay",
        size="VehicleCount", color="VehicleCount",
        zoom=4, color_continuous_scale="Inferno"
    )
    fig_anim.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_anim, use_container_width=True)

# =========================================================
# TAB 3 — ML PREDICTIONS
# =========================================================
with tab3:

    st.header("🤖 ML Predictions")

    # RANDOM FOREST
    st.subheader("🌲 RandomForest Prediction")
    ml = df[["HourOfDay", "DayOfWeek", "VehicleCount"]].copy()
    ml["DayOfWeek"] = ml["DayOfWeek"].astype("category").cat.codes
    X, y = ml[["HourOfDay", "DayOfWeek"]], ml["VehicleCount"]

    rf = RandomForestRegressor()
    rf.fit(X, y)

    hr = st.slider("Hour", 0, 23)
    dy = st.selectbox("Day", sorted(df["DayOfWeek"].unique()))
    dy_num = pd.Categorical([dy], categories=sorted(df["DayOfWeek"].unique())).codes[0]

    rf_pred = rf.predict([[hr, dy_num]])[0]
    st.success(f"Predicted Vehicles: {int(rf_pred)}")

    # LSTM MODEL
    st.subheader("📈 LSTM Forecast")

    seq = df_filtered.sort_values("Timestamp")["VehicleCount"].values.astype(float)
    window = 24
    X_lstm, y_lstm = [], []

    for i in range(len(seq) - window):
        X_lstm.append(seq[i:i+window])
        y_lstm.append(seq[i+window])

    X_lstm = torch.tensor(np.array(X_lstm).reshape(-1, window, 1), dtype=torch.float32)
    y_lstm = torch.tensor(np.array(y_lstm), dtype=torch.float32)

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = LSTMModel()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(6):
        optim.zero_grad()
        out = model(X_lstm).squeeze()
        loss = loss_fn(out, y_lstm)
        loss.backward()
        optim.step()

    last_seq = torch.tensor(seq[-window:].reshape(1, window, 1), dtype=torch.float32)
    lstm_pred = model(last_seq).item()

    st.info(f"🔮 LSTM Next-Hour Forecast: {int(lstm_pred)} vehicles")

# =========================================================
# TAB 4 — UPLOAD
# =========================================================
with tab4:
    st.header("📁 Upload Custom Data")
    if uploaded_file:
        st.success("Custom dataset loaded.")
    else:
        st.warning("Using default dataset.")
