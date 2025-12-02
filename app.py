import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Simulated Traffic Dashboard", layout="wide")

# ===============================
# Load Data
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv("simulated_traffic_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

df = load_data()

# ===============================
# Sidebar Filters
# ===============================

st.sidebar.header("Filter Options")

# Date range filter
min_date = df["Timestamp"].min().date()
max_date = df["Timestamp"].max().date()

date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

start_date, end_date = date_range
df = df[(df["Timestamp"].dt.date >= start_date) & (df["Timestamp"].dt.date <= end_date)]

# State filter
selected_state = st.sidebar.selectbox(
    "Select State",
    sorted(df["State"].unique())
)

# Filter cities in state
cities_in_state = df[df["State"] == selected_state]["City"].unique()

# City filter
selected_city = st.sidebar.selectbox(
    "Select City",
    ["All Cities"] + list(cities_in_state)
)

if selected_city == "All Cities":
    filtered_df = df[df["State"] == selected_state]
else:
    filtered_df = df[(df["State"] == selected_state) & (df["City"] == selected_city)]

# ===============================
# State-Level Summary Cards
# ===============================

st.header(f"🚦 Traffic Dashboard – {selected_state}")

col1, col2, col3, col4 = st.columns(4)

avg_traffic = filtered_df["VehicleCount"].mean()
peak_hour = filtered_df.groupby("HourOfDay")["VehicleCount"].mean().idxmax()
max_traffic = filtered_df["VehicleCount"].max()
total_records = len(filtered_df)

col1.metric("Avg Traffic", f"{avg_traffic:.1f}")
col2.metric("Peak Hour", f"{peak_hour}:00")
col3.metric("Max Vehicle Count", max_traffic)
col4.metric("Records Shown", total_records)

# ===============================
# Traffic Over Time Line Chart
# ===============================

st.subheader("📈 Traffic Over Time")

fig1 = px.line(
    filtered_df,
    x='Timestamp',
    y='VehicleCount',
    color='City',
    title=f"Traffic Over Time – {selected_state} ({selected_city})"
)

st.plotly_chart(fig1, use_container_width=True)

# ===============================
# Hourly Heatmap
# ===============================

st.subheader("🔥 Peak Hour Heatmap")

heatmap_data = filtered_df.groupby(["DayOfWeek", "HourOfDay"])["VehicleCount"].mean().reset_index()

fig_heat = px.density_heatmap(
    heatmap_data,
    x="HourOfDay",
    y="DayOfWeek",
    z="VehicleCount",
    color_continuous_scale="Inferno",
    title="Heatmap of Traffic Volume by Day & Hour"
)

st.plotly_chart(fig_heat, use_container_width=True)

# ===============================
# Compare Two Cities
# ===============================

st.subheader("🏙 Compare Two Cities")

city_list = list(cities_in_state)
city1 = st.selectbox("City 1", city_list, index=0)
city2 = st.selectbox("City 2", city_list, index=1)

df_compare = df[(df["City"].isin([city1, city2]))]

fig_compare = px.line(
    df_compare,
    x="Timestamp",
    y="VehicleCount",
    color="City",
    title=f"Traffic Comparison: {city1} vs {city2}"
)

st.plotly_chart(fig_compare, use_container_width=True)

# ===============================
# Interactive Map (City Points)
# ===============================

st.subheader("🗺 Interactive Map")

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

df["Lat"] = df["City"].map(lambda c: city_coords[c][0])
df["Lon"] = df["City"].map(lambda c: city_coords[c][1])

fig_map = px.scatter_mapbox(
    df.drop_duplicates("City"),
    lat="Lat",
    lon="Lon",
    hover_name="City",
    zoom=5,
    height=500
)

fig_map.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig_map, use_container_width=True)

# ===============================
# Export Filtered Data
# ===============================

st.subheader("⬇ Export Filtered Results")

st.download_button(
    label="Download Filtered CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_traffic_data.csv",
    mime="text/csv"
)

# ===============================
# Simple ML Prediction
# ===============================

st.subheader("🤖 Predict Traffic Volume (Simple ML Model)")

# Prepare ML dataset
ml_df = df[["HourOfDay", "DayOfWeek", "VehicleCount"]].copy()
ml_df["DayOfWeek"] = ml_df["DayOfWeek"].astype("category").cat.codes

X = ml_df[["HourOfDay", "DayOfWeek"]]
y = ml_df["VehicleCount"]

model = RandomForestRegressor()
model.fit(X, y)

# User Input
pred_hour = st.slider("Select Hour of Day", 0, 23, 8)
pred_day = st.selectbox("Select Day", sorted(df["DayOfWeek"].unique()))
pred_day_num = pd.Categorical([pred_day], categories=sorted(df["DayOfWeek"].unique())).codes[0]

prediction = model.predict([[pred_hour, pred_day_num]])[0]

st.success(f"Predicted Vehicle Count: **{int(prediction)}**")
