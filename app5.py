import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    return pd.read_excel("Final_dataset_filled.xlsx")

df = load_data()

# Mapping of parameters
params = {
    "Temperature (°C)": ("Temp", "T(°C)"),
    "pH": ("pH", "PROBE_PH"),
    "Conductivity (µS/cm)": ("CondSp", "SpCOND(uS/cm)"),
    "Dissolved Oxygen (mg/l)": ("ODO Conc", "ODO(mg/l)")
}

st.title("📊 Comparison of Vertical Profiles vs Boat Measurements")

# Sidebar selections
station_ids = sorted(df["Station"].dropna().unique())
selected_station = st.sidebar.selectbox("🛰️ Select a Station", station_ids)
selected_param = st.sidebar.selectbox("📈 Select a Parameter", list(params.keys()))
col_vp, col_boat = params[selected_param]

# Filter data for selected station
station_df = df[df["Station"] == selected_station].copy()

# Drop rows with missing necessary values
station_df = station_df[
    station_df[col_vp].notna() &
    station_df[col_boat].notna() &
    station_df["Profondeur"].notna() &
    station_df["SAMPLE_DEPTH"].notna()
]

# Get candidate depths that have matching SAMPLE_DEPTH within ±0.03
depth_candidates = station_df["Profondeur"].unique()
valid_depths = []

for depth in sorted(depth_candidates):
    group = station_df[
        ((station_df["Profondeur"] - depth).abs() <= 0.03) &
        ((station_df["SAMPLE_DEPTH"] - depth).abs() <= 0.03)
    ]
    if not group.empty:
        valid_depths.append(depth)

# Choose 3 depths spread out
selected_depths = []
if len(valid_depths) >= 3:
    valid_depths = sorted(valid_depths)
    selected_depths = [valid_depths[0], valid_depths[len(valid_depths)//2], valid_depths[-1]]
else:
    st.warning("Not enough valid depths for this station.")
    st.stop()

# Prepare data for grouped bar chart
plot_data = {
    "Depth": [],
    "Type": [],
    "Value": [],
    "Date": []
}

for depth in selected_depths:
    group = station_df[
        ((station_df["Profondeur"] - depth).abs() <= 0.03) &
        ((station_df["SAMPLE_DEPTH"] - depth).abs() <= 0.03)
    ]

    if group.empty:
        continue

    # Take mean value for simplicity
    vp_mean = group[col_vp].mean()
    boat_mean = group[col_boat].mean()
    dates = pd.to_datetime(group["Date"]).dt.strftime('%Y-%m-%d').unique()
    date_str = ", ".join(dates)

    plot_data["Depth"].extend([f"{depth:.2f} m"] * 2)
    plot_data["Type"].extend(["VP", "Boat"])
    plot_data["Value"].extend([vp_mean, boat_mean])
    plot_data["Date"].extend([date_str] * 2)

# Create grouped bar chart
fig = go.Figure()

depths_unique = list(set(plot_data["Depth"]))
for depth in depths_unique:
    for t in ["VP", "Boat"]:
        i = (np.array(plot_data["Depth"]) == depth) & (np.array(plot_data["Type"]) == t)
        val = np.array(plot_data["Value"])[i]
        date = np.array(plot_data["Date"])[i]
        fig.add_trace(go.Bar(
            x=[t],
            y=val,
            name=f"{t} - {depth}",
            text=date,
            textposition='outside',
        ))

fig.update_layout(
    barmode='group',
    title=f"{selected_param} Comparison - Station {selected_station}",
    xaxis_title="Measurement Type",
    yaxis_title=f"{selected_param}",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ======================
# Station Map
# ======================

st.subheader("🗺️ Station Map")

station_map = df[["Station", "Latitude", "Longitude"]].dropna()
station_map = station_map.groupby("Station").first().reset_index()
station_map = station_map.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=station_map["latitude"].mean(),
        longitude=station_map["longitude"].mean(),
        zoom=10,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=station_map,
            get_position='[longitude, latitude]',
            get_fill_color='[0, 100, 255, 160]',
            get_radius=1000,  # Reduced size here
            pickable=True,
        )
    ],
    tooltip={"text": "Station: {Station}"}
))
