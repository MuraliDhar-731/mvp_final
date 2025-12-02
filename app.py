import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Advanced Traffic Dashboard", layout="wide")

# ===============================================================
# LOAD DATA + UPLOAD SUPPORT
# ===============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("simulated_traffic_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

# Drag & Drop Uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
else:
    df = load_data()

# ===============================================================
# TAB LAYOUT
# ===============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🗺 Maps",
    "🤖 ML Predictions",
    "📁 Upload Data"
])

# ===============================================================
# COMMON FILTERS USED IN MULTIPLE TABS
# ===============================================================

states = sorted(df["State"].unique())

selected_state = st.sidebar.selectbox("Select State", states)

cities = sorted(df[df["State"] == selected_state]["City"].unique())
selected_city = st.sidebar.selectbox("Select City", ["All Cities"] + cities)

df_filtered = df[df["State"] == selected_state]
if selected_city != "All Cities":
    df_filtered = df_filtered[df_filtered["City"] == selected_city]

# ===============================================================
# TAB 1 – DASHBOARD
# ===============================================================

with tab1:

    st.header(f"📊 Traffic Dashboard – {selected_state}")

    # KPI CARDS
    avg_traffic = df_filtered["VehicleCount"].mean()
    peak_hour = df_filtered.groupby("HourOfDay")["VehicleCount"].mean().idxmax()
    max_traffic = df_filtered["VehicleCount"].max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Traffic", f"{avg_traffic:.1f}")
    col2.metric("Peak Hour", f"{peak_hour}:00")
    col3.metric("Max Vehicle Count", max_traffic)

    # Line Graph
    st.subheader("📈 Traffic Over Time")
    fig1 = px.line(
        df_filtered,
        x="Timestamp",
        y="VehicleCount",
        color="City",
        title=f"Traffic Over Time – {selected_state} ({selected_city})"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Heatmap
    st.subheader("🔥 Peak Hour Heatmap")
    heatmap_data = df_filtered.groupby(["DayOfWeek", "HourOfDay"])["VehicleCount"].mean().reset_index()

    fig_heat = px.density_heatmap(
        heatmap_data, x="HourOfDay", y="DayOfWeek", z="VehicleCount",
        color_continuous_scale="Inferno"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ===============================================================
# TAB 2 – MAPS
# ===============================================================

with tab2:

    st.header("🗺 Interactive Maps")

    # NYC-centric map
    st.subheader("🗽 Full NYC Zoom Map")
    
    city_coords = {
        "New York City": (40.7128, -74.0060),
        "Buffalo": (42.8864, -78.8784),
        "Rochester": (43.1566, -77.6088),
        "Newark": (40.7357, -74.1724),
        "Jersey City": (40.7178, -74.0431),
        "Paterson": (40.9168, -74.1718),
    }

    df_map = df.drop_duplicates("City").copy()
    df_map["Lat"] = df_map["City"].map(lambda x: city_coords.get(x, (0, 0))[0])
    df_map["Lon"] = df_map["City"].map(lambda x: city_coords.get(x, (0, 0))[1])

    fig_map = px.scatter_mapbox(
        df_map,
        lat="Lat",
        lon="Lon",
        color="VehicleCount",
        size="VehicleCount",
        hover_name="City",
        color_continuous_scale="Turbo",
        zoom=6
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

    # Animated Map
    st.subheader("🎞 Animated Traffic Map (Time-Lapse)")

    fig_anim = px.scatter_mapbox(
        df,
        lat="Lat",
        lon="Lon",
        color="VehicleCount",
        size="VehicleCount",
        hover_name="City",
        color_continuous_scale="Inferno",
        animation_frame="HourOfDay",
        zoom=4,
        height=650
    )
    fig_anim.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_anim, use_container_width=True)

# ===============================================================
# TAB 3 – ML PREDICTIONS
# ===============================================================

with tab3:

    st.header("🤖 Machine Learning Models")

    # RandomForest Model
    st.subheader("🌲 RandomForest Prediction")

    ml_df = df[["HourOfDay", "DayOfWeek", "VehicleCount"]].copy()
    ml_df["DayOfWeek"] = ml_df["DayOfWeek"].astype("category").cat.codes

    X = ml_df[["HourOfDay", "DayOfWeek"]]
    y = ml_df["VehicleCount"]

    rf = RandomForestRegressor()
    rf.fit(X, y)

    hour = st.slider("Hour of Day", 0, 23)
    day = st.selectbox("Day of Week", sorted(df["DayOfWeek"].unique()))
    day_num = pd.Categorical([day], categories=sorted(df["DayOfWeek"].unique())).codes[0]

    rf_pred = rf.predict([[hour, day_num]])[0]
    st.success(f"RandomForest Prediction: {int(rf_pred)} vehicles")

    # LSTM Model
    st.subheader("📈 LSTM Forecast")

    # Prepare sequence data
    seq_df = df_filtered.sort_values("Timestamp")
    series = seq_df["VehicleCount"].values.astype(float)

    window = 24
    X_lstm, y_lstm = [], []

    for i in range(len(series) - window):
        X_lstm.append(series[i:i+window])
        y_lstm.append(series[i+window])

    X_lstm = np.array(X_lstm).reshape(-1, window, 1)
    y_lstm = np.array(y_lstm)

    # Build LSTM model
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(window, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_lstm, y_lstm, epochs=4, batch_size=16, verbose=0)

    # Predict next hour
    last_seq = series[-window:].reshape(1, window, 1)
    lstm_pred = model.predict(last_seq)[0][0]

    st.info(f"LSTM Forecast for Next Hour: **{int(lstm_pred)} vehicles**")

# ===============================================================
# TAB 4 – UPLOAD NEW DATA
# ===============================================================

with tab4:
    st.header("📁 Upload Custom Traffic Data")
    st.write("Upload a CSV to override the dataset used in all dashboards.")
    st.write("Columns required: **State, City, Timestamp, HourOfDay, DayOfWeek, VehicleCount**")
    
    if uploaded_file:
        st.success("Custom dataset loaded successfully!")
    else:
        st.warning("Using default dataset.")
