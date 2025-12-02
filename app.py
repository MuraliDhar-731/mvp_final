import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

# =========================================================
# PAGE CONFIG + THEME
# =========================================================

st.set_page_config(page_title="Traffic Lens AI", page_icon="🚦", layout="wide")

PRIMARY = "#FF4B4B"
SECONDARY = "#1E1E1E"
ACCENT = "#FFD700"
BG_COLOR = "#F5F5F5"
TEXT_COLOR = "#222222"

# Smooth fade animation
st.markdown("""
<style>
@keyframes fadeSmooth {
    from { opacity:0; transform:translateY(10px); }
    to { opacity:1; transform:translateY(0px); }
}
div { animation: fadeSmooth 0.8s ease-in-out; }
body { background-color: #F5F5F5; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA + FILE UPLOAD
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
# HERO HEADER
# =========================================================

hero_html = f"""
<div style="
    padding: 25px;
    background-color: {PRIMARY};
    border-radius: 14px;
    margin-bottom: 30px;
    text-align: center;
">

<h1 style="font-size: 55px; color: white; font-weight: 900; margin-bottom: 5px;">
🚦 Traffic Lens AI
</h1>

<h3 style="color:white; font-weight:600; opacity:0.9; margin-top:-5px;">
🔧 Powered by MID LABS
</h3>

<p id="typing" style="font-size: 22px; color: white; font-weight: 500;"></p>

<script>
let text = "A Smarter Way to See Traffic.";
let index = 0;
function type() {{
    if (index < text.length) {{
        document.getElementById("typing").innerHTML += text.charAt(index);
        index++;
        setTimeout(type, 55);
    }}
}}
type();
</script>

</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; margin-bottom:20px;'>
    <img src='https://img.icons8.com/external-flatarticons-blue-flatarticons/512/external-traffic-smart-city-flatarticons-blue-flatarticons.png'
         width='120'>
</div>
""", unsafe_allow_html=True)

# =========================================================
# FILTERS
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
# TAB 1 – DASHBOARD
# =========================================================
with tab1:

    st.header(f"📊 Traffic Overview – {selected_state}")

    avg_traffic = df_filtered["VehicleCount"].mean()
    peak_hour = df_filtered.groupby("HourOfDay")["VehicleCount"].mean().idxmax()
    max_traffic = df_filtered["VehicleCount"].max()

    kpi_html = f"""
    <div style="display:flex; gap:20px; justify-content:center; margin-bottom:25px">

        <div style="flex:1; background:white; padding:20px; border-left:8px solid {PRIMARY};
            border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.08);">
            <h3 style="margin:0; color:{PRIMARY};">Average Traffic</h3>
            <p style="font-size:26px; font-weight:700;">{avg_traffic:.1f}</p>
        </div>

        <div style="flex:1; background:white; padding:20px; border-left:8px solid {ACCENT};
            border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.08);">
            <h3 style="margin:0; color:{ACCENT};">Peak Hour</h3>
            <p style="font-size:26px; font-weight:700;">{peak_hour}:00</p>
        </div>

        <div style="flex:1; background:white; padding:20px; border-left:8px solid {SECONDARY};
            border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.08);">
            <h3 style="margin:0; color:{SECONDARY};">Max Vehicles</h3>
            <p style="font-size:26px; font-weight:700;">{max_traffic}</p>
        </div>

    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.subheader("📈 Traffic Over Time")
    st.plotly_chart(px.line(df_filtered, x="Timestamp", y="VehicleCount", color="City"),
                    use_container_width=True)

    st.subheader("🔥 Heatmap")
    heatmap_df = df_filtered.groupby(["DayOfWeek","HourOfDay"])["VehicleCount"].mean().reset_index()
    st.plotly_chart(px.density_heatmap(heatmap_df, x="HourOfDay", y="DayOfWeek",
                                       z="VehicleCount", color_continuous_scale="Inferno"),
                    use_container_width=True)

    st.subheader("🏙 Compare Two Cities")
    c1 = st.selectbox("City 1", cities)
    c2 = st.selectbox("City 2", cities, index=1)
    comp = df[df["City"].isin([c1, c2])]
    st.plotly_chart(px.line(comp, x="Timestamp", y="VehicleCount", color="City"),
                    use_container_width=True)

# =========================================================
# TAB 2 – MAPS
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

    st.subheader("🗽 NYC Zoom Map")
    st.plotly_chart(px.scatter_mapbox(df_map, lat="Lat", lon="Lon",
                                      size="VehicleCount", color="VehicleCount",
                                      zoom=6, color_continuous_scale="Turbo")
                    .update_layout(mapbox_style="open-street-map"), use_container_width=True)

    st.subheader("🎞 Animated Traffic Map")
    st.plotly_chart(px.scatter_mapbox(df_map, lat="Lat", lon="Lon",
                                      animation_frame="HourOfDay",
                                      size="VehicleCount", color="VehicleCount",
                                      zoom=4, color_continuous_scale="Inferno")
                    .update_layout(mapbox_style="open-street-map"), use_container_width=True)

# =========================================================
# TAB 3 – ML PREDICTIONS
# =========================================================
with tab3:

    st.header("🤖 ML Predictions")

    st.subheader("🌲 RandomForest Prediction")

    ml = df[["HourOfDay","DayOfWeek","VehicleCount"]].copy()
    ml["DayOfWeek"] = ml["DayOfWeek"].astype("category").cat.codes

    X = ml[["HourOfDay","DayOfWeek"]]
    y = ml["VehicleCount"]

    rf = RandomForestRegressor()
    rf.fit(X, y)

    hr = st.slider("Hour", 0, 23)
    dy = st.selectbox("Day", sorted(df["DayOfWeek"].unique()))
    dy_num = pd.Categorical([dy], categories=sorted(df["DayOfWeek"].unique())).codes[0]

    st.success(f"RandomForest Prediction: {int(rf.predict([[hr, dy_num]])[0])} vehicles")

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

    for _ in range(5):
        optim.zero_grad()
        out = model(X_lstm).squeeze()
        loss = loss_fn(out, y_lstm)
        loss.backward()
        optim.step()

    last = torch.tensor(seq[-window:].reshape(1, window, 1), dtype=torch.float32)
    lstm_pred = model(last).item()

    st.info(f"🔮 LSTM Forecast: {int(lstm_pred)} vehicles next hour")

# =========================================================
# TAB 4 – UPLOAD
# =========================================================
with tab4:
    st.header("📁 Upload Custom Data")
    if uploaded_file:
        st.success("Custom dataset loaded.")
    else:
        st.warning("Using default dataset.")

# =========================================================
# PREMIUM FOOTER — MID LABS
# =========================================================

footer = """
<h2 style='font-weight:900; letter-spacing:1.2px; margin-bottom:25px; text-align:center;'>
    🚀 MID LABS – LEADERSHIP TEAM
</h2>

<div style='display:flex; justify-content:center; gap:50px; flex-wrap:wrap;'>

    <div style='text-align:center;'>
        <img src='https://img.icons8.com/ios-filled/256/user-male-circle.png'
             style='width:110px; height:110px; border-radius:50%; border:3px solid #FFD700;
             box-shadow:0px 4px 10px rgba(0,0,0,0.2);'>
        <p style='font-size:22px; font-weight:800;'>Muralidhar</p>
        <p style='opacity:0.85; font-size:16px;'>CEO & Founder</p>
    </div>

    <div style='text-align:center;'>
        <img src='https://img.icons8.com/ios-filled/256/user-female-circle.png'
             style='width:110px; height:110px; border-radius:50%; border:3px solid #FFD700;
             box-shadow:0px 4px 10px rgba(0,0,0,0.2);'>
        <p style='font-size:22px; font-weight:800;'>Ishika</p>
        <p style='opacity:0.85; font-size:16px;'>Chief Marketing Officer (CMO)</p>
    </div>

    <div style='text-align:center;'>
        <img src='https://img.icons8.com/ios-filled/256/user.png'
             style='width:110px; height:110px; border-radius:50%; border:3px solid #FFD700;
             box-shadow:0px 4px 10px rgba(0,0,0,0.2);'>
        <p style='font-size:22px; font-weight:800;'>Devakinandan</p>
        <p style='opacity:0.85; font-size:16px;'>Chief Innovation Officer (CIO)</p>
    </div>

</div>

<div style='margin:35px 0 15px; text-align:center;'>
    <a href='#' style='margin:0 15px;'><img src='https://img.icons8.com/ios-filled/50/linkedin.png' width='32'></a>
    <a href='#' style='margin:0 15px;'><img src='https://img.icons8.com/ios-filled/50/instagram-new.png' width='32'></a>
    <a href='#' style='margin:0 15px;'><img src='https://img.icons8.com/ios-filled/50/domain.png' width='32'></a>
</div>

<p style='font-size:16px; opacity:0.9; text-align:center;'>
    Made with ❤️ by <strong>MID LABS</strong>
</p>

<p style='font-size:13px; opacity:0.75; text-align:center;'>
    © 2025 MID LABS. All rights reserved.
</p>
"""

st.markdown(footer, unsafe_allow_html=True)
