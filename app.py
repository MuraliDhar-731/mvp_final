import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

st.set_page_config(page_title="Advanced Traffic Dashboard", layout="wide")


# =========================================================
# LOAD DATA (Supports Drag & Drop)
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
# COMMON FILTERS
# =========================================================

st.sidebar.subheader("Filters")

# Date range filter
min_date = df["Timestamp"].min().date()
max_date = df["Timestamp"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

start_date, end_date = date_range
df = df[(df["Timestamp"].dt.date >= start_date) & (df["Timestamp"].dt.date <= end_date)]

# State filter
states = sorted(df["State"].unique())
selected_state = st.sidebar.selectbox("State", states)

# City filter
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

    st.header(f"📊 Traffic Dashboard – {selected_state}")

    # KPI Metrics
    avg_traffic = df_filtered["VehicleCount"].mean()
    peak_hour = df_filtered.groupby("HourOfDay")["VehicleCount"].mean().idxmax()
    max_traffic = df_filtered["VehicleCount"].max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Traffic", f"{avg_traffic:.1f}")
    col2.metric("Peak Hour", f"{peak_hour}:00")
    col3.metric("Maximum Traffic", max_traffic)

    # Line Chart
    st.subheader("📈 Traffic Over Time")
    fig1 = px.line(
        df_filtered,
        x="Timestamp",
        y="VehicleCount",
        color="City",
        title=f"Traffic Over Time – {selected_state}"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Heatmap
    st.subheader("🔥 Peak Hour Heatmap")
    heatmap_df = df_filtered.groupby(["DayOfWeek","HourOfDay"])["VehicleCount"].mean().reset_index()
    fig_heat = px.density_heatmap(
        heatmap_df,
        x="HourOfDay",
        y="DayOfWeek",
        z="VehicleCount",
        color_continuous_scale="Inferno"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Compare 2 cities
    st.subheader("🏙 Compare Two Cities")
    compare_city1 = st.selectbox("City 1", cities)
    compare_city2 = st.selectbox("City 2", cities, index=1)

    comp_df = df[df["City"].isin([compare_city1, compare_city2])]

    fig_compare = px.line(
        comp_df,
        x="Timestamp",
        y="VehicleCount",
        color="City",
        title=f"Comparison: {compare_city1} vs {compare_city2}"
    )
    st.plotly_chart(fig_compare, use_container_width=True)


# =========================================================
# TAB 2 — MAPS
# =========================================================
with tab2:

    st.header("🗺 Geographical Traffic Maps")

    # Coordinates
    city_coords = {
        "Hartford": (41.7658, -72.6734),
        "New Haven": (41.3083, -72.9279),
        "Stamford": (41.0534, -73.5387),
        "Boston": (42.3601, -71.0589),
        "Worcester": (42.2626, -71.8023),
        "Springfield": (42.1015, -72.5898),
        "Newark": (40.7357, -74.1724),
        "Jersey City": (40.7178, -74.0431),
        "Paterson": (40.9168, -74.1718),
        "New York City": (40.7128, -74.0060),
        "Buffalo": (42.8864, -78.8784),
        "Rochester": (43.1566, -77.6088),
    }

    df_map = df.copy()
    df_map["Lat"] = df_map["City"].map(lambda x: city_coords[x][0])
    df_map["Lon"] = df_map["City"].map(lambda x: city_coords[x][1])

    # NYC Zoom Map
    st.subheader("🗽 NYC Zoom-Level Traffic Map")

    fig_nyc = px.scatter_mapbox(
        df_map,
        lat="Lat",
        lon="Lon",
        size="VehicleCount",
        color="VehicleCount",
        hover_name="City",
        zoom=6,
        color_continuous_scale="Turbo",
        height=600
    )
    fig_nyc.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_nyc, use_container_width=True)

    # Animated Map
    st.subheader("🎞 Animated Traffic Map")

    fig_anim = px.scatter_mapbox(
        df_map,
        lat="Lat",
        lon="Lon",
        size="VehicleCount",
        color="VehicleCount",
        animation_frame="HourOfDay",
        hover_name="City",
        zoom=4,
        color_continuous_scale="Inferno",
        height=650
    )
    fig_anim.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_anim, use_container_width=True)


# =========================================================
# TAB 3 — ML PREDICTIONS (RandomForest + PyTorch LSTM)
# =========================================================
with tab3:

    st.header("🤖 ML Predictions")

    # -----------------------------
    # RandomForest
    # -----------------------------
    st.subheader("🌲 RandomForest Prediction")

    ml_df = df[["HourOfDay","DayOfWeek","VehicleCount"]].copy()
    ml_df["DayOfWeek"] = ml_df["DayOfWeek"].astype("category").cat.codes

    X = ml_df[["HourOfDay","DayOfWeek"]]
    y = ml_df["VehicleCount"]

    rf = RandomForestRegressor()
    rf.fit(X, y)

    pred_hour = st.slider("Hour of Day", 0, 23)
    pred_day = st.selectbox("Day of Week", sorted(df["DayOfWeek"].unique()))
    pred_day_num = pd.Categorical([pred_day], categories=sorted(df["DayOfWeek"].unique())).codes[0]

    rf_result = rf.predict([[pred_hour, pred_day_num]])[0]
    st.success(f"RandomForest Prediction: **{int(rf_result)} vehicles**")


    # -----------------------------
    # PyTorch LSTM MODEL
    # -----------------------------
    st.subheader("📈 PyTorch LSTM Traffic Forecasting")

    seq_df = df_filtered.sort_values("Timestamp")
    series = seq_df["VehicleCount"].values.astype(float)

    # create sequences
    window = 24
    X_lstm, y_lstm = [], []

    for i in range(len(series) - window):
        X_lstm.append(series[i:i+window])
        y_lstm.append(series[i+window])

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)

    # reshape for LSTM: (batch, seq, features)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], window, 1)

    # convert to tensors
    X_tensor = torch.tensor(X_lstm, dtype=torch.float32)
    y_tensor = torch.tensor(y_lstm, dtype=torch.float32)

    # Define LSTM model
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train LSTM (light training for Streamlit)
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output.squeeze(), y_tensor)
        loss.backward()
        optimizer.step()

    # Predict next hour
    last_seq = torch.tensor(series[-window:].reshape(1, window, 1), dtype=torch.float32)
    lstm_pred = model(last_seq).item()

    st.info(f"🔮 LSTM Forecast (Next Hour): **{int(lstm_pred)} vehicles**")


# =========================================================
# TAB 4 — UPLOAD
# =========================================================
with tab4:
    st.header("📁 Upload Custom Traffic Data")
    st.write("Upload a CSV file to override the dataset for all tabs.")
    st.write("Columns required: *State, City, Timestamp, HourOfDay, DayOfWeek, VehicleCount*")

    if uploaded_file:
        st.success("Custom dataset loaded successfully!")
    else:
        st.warning("Using default dataset.")
