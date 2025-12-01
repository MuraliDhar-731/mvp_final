import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Simulated Traffic Dashboard", layout="wide")

st.title("🚦 Simulated Traffic Data Dashboard")
st.write("Interactive dashboard showing traffic patterns for CT, MA, NJ, NY.")

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("simulated_traffic_data.csv")

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options")
states = st.sidebar.multiselect("Select States", df["State"].unique(), df["State"].unique())
cities = st.sidebar.multiselect("Select Cities", df["City"].unique(), df["City"].unique())

filtered_df = df[df["State"].isin(states) & df["City"].isin(cities)]

# User View – Plot 1
st.subheader("📈 Live Traffic Over Time")
fig1 = px.line(
    filtered_df,
    x='Timestamp',
    y='VehicleCount',
    color='City',
    title='Live Traffic Over Time'
)
st.plotly_chart(fig1, use_container_width=True)

# User View – Plot 2
st.subheader("⏱ Average Hourly Traffic Patterns")
df_hourly = filtered_df.groupby(['DayOfWeek', 'HourOfDay'], as_index=False)['VehicleCount'].mean()

fig2 = px.line(
    df_hourly,
    x='HourOfDay',
    y='VehicleCount',
    color='DayOfWeek',
    title='Hourly Traffic Pattern (Weekday vs Weekend)'
)
st.plotly_chart(fig2, use_container_width=True)

# Admin View
st.subheader("🗂 Admin View – Raw Traffic Data")
st.dataframe(filtered_df.head(20))

st.download_button(
    label="Download Full Traffic CSV",
    data=df.to_csv(index=False),
    file_name="simulated_traffic_data.csv",
    mime="text/csv"
)
