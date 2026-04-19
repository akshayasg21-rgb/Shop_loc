import streamlit as st
import osmnx as ox
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.neighbors import BallTree

st.set_page_config(layout="wide")

st.title("📍 Shop Location Planner — Chennai")

st.write("Find promising locations for opening a shop based on nearby infrastructure and competition.")

# ---------------------------------------------------
# AREAS IN CHENNAI
# ---------------------------------------------------

areas = {
    "T Nagar":"T Nagar Chennai India",
    "Adyar":"Adyar Chennai India",
    "Velachery":"Velachery Chennai India",
    "Anna Nagar":"Anna Nagar Chennai India",
    "Tambaram":"Tambaram Chennai India",
    "Guindy":"Guindy Chennai India"
}

area = st.selectbox("Select Area",list(areas.keys()))

shop_type = st.selectbox(
    "Select Shop Type",
    ["flower_shop","restaurant","grocery","clothing"]
)

SEARCH_RADIUS = 800


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data(place):

    lat,lon = ox.geocode(place)

    tags = {
        "amenity":[
            "school",
            "college",
            "hospital",
            "restaurant",
            "bank",
            "police"
        ],
        "tourism":["hotel"],
        "shop":True
    }

    gdf = ox.features_from_point(
        (lat,lon),
        tags=tags,
        dist=SEARCH_RADIUS
    )

    gdf = gdf[['amenity','shop','tourism','name','geometry']]
    gdf = gdf.dropna(subset=['geometry'])

    gdf = gdf.to_crs(epsg=3857)
    gdf['geometry'] = gdf.geometry.centroid
    gdf = gdf.to_crs(epsg=4326)

    gdf['lat'] = gdf.geometry.y
    gdf['lon'] = gdf.geometry.x

    return lat,lon,gdf.reset_index(drop=True)


# ---------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------

if st.button("Analyze Area"):
    st.session_state.run_analysis = True

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

if st.session_state.run_analysis:

    place = areas[area]

    lat,lon,gdf = load_data(place)

# ---------------------------------------------------
# SPLIT DATA
# ---------------------------------------------------

    schools = gdf[gdf['amenity']=="school"]
    colleges = gdf[gdf['amenity']=="college"]
    hospitals = gdf[gdf['amenity']=="hospital"]
    restaurants = gdf[gdf['amenity']=="restaurant"]
    banks = gdf[gdf['amenity']=="bank"]
    police = gdf[gdf['amenity']=="police"]
    hotels = gdf[gdf['tourism']=="hotel"]
    shops = gdf[gdf['shop'].notna()]

# ---------------------------------------------------
# COMPETITORS
# ---------------------------------------------------

    if shop_type=="flower_shop":
        competitors = shops[shops['shop']=="florist"]

    elif shop_type=="restaurant":
        competitors = restaurants

    elif shop_type=="grocery":
        competitors = shops[shops['shop']=="supermarket"]

    else:
        competitors = shops


# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

    st.subheader("Nearby Infrastructure")

    c1,c2,c3 = st.columns(3)

    c1.metric("Schools",len(schools))
    c1.metric("Colleges",len(colleges))

    c2.metric("Hospitals",len(hospitals))
    c2.metric("Hotels",len(hotels))

    c3.metric("Banks",len(banks))
    c3.metric("Competitors",len(competitors))


# ---------------------------------------------------
# SHOW COMPETITORS
# ---------------------------------------------------

    st.subheader("Nearby Competitors")

    competitor_list = competitors[['name','lat','lon']].dropna()

    if len(competitor_list)>0:

        competitor_list = competitor_list.head(10)

        for _,row in competitor_list.iterrows():

            name = row['name'] if pd.notna(row['name']) else "Unnamed Shop"

            st.write(f"• {name}")

    else:

        st.write("No major competitors nearby.")


# ---------------------------------------------------
# BUILD TREES
# ---------------------------------------------------

    def build_tree(df):

        if len(df)==0:
            return None

        return BallTree(
            np.radians(df[['lat','lon']]),
            metric="haversine"
        )

    school_tree = build_tree(schools)
    college_tree = build_tree(colleges)
    hospital_tree = build_tree(hospitals)
    hotel_tree = build_tree(hotels)
    comp_tree = build_tree(competitors)


# ---------------------------------------------------
# CREATE SEARCH GRID
# ---------------------------------------------------

    grid = []

    for i in np.linspace(lat-0.008,lat+0.008,20):
        for j in np.linspace(lon-0.008,lon+0.008,20):
            grid.append([i,j])

    grid = pd.DataFrame(grid,columns=["lat","lon"])

    coords = np.radians(grid[['lat','lon']])

    r = 0.4/6371


    def count(tree):

        if tree is None:
            return np.zeros(len(coords))

        return tree.query_radius(coords,r=r,count_only=True)


    grid['schools']=count(school_tree)
    grid['colleges']=count(college_tree)
    grid['hospitals']=count(hospital_tree)
    grid['hotels']=count(hotel_tree)
    grid['competition']=count(comp_tree)


# ---------------------------------------------------
# SCORE LOCATIONS
# ---------------------------------------------------

    if shop_type=="flower_shop":

        grid['score'] = (
            grid['hospitals']*0.5 +
            grid['schools']*0.2 -
            grid['competition']*0.6
        )

    elif shop_type=="restaurant":

        grid['score'] = (
            grid['colleges']*0.4 +
            grid['hotels']*0.4 -
            grid['competition']*0.5
        )

    elif shop_type=="grocery":

        grid['score'] = (
            grid['schools']*0.3 +
            grid['colleges']*0.3 -
            grid['competition']*0.4
        )

    else:

        grid['score'] = (
            grid['hotels']*0.3 -
            grid['competition']*0.5
        )

    best_locations = grid.sort_values("score",ascending=False).head(5)


# ---------------------------------------------------
# LOCATION REASONS
# ---------------------------------------------------

    def generate_reason(row):

        reasons=[]

        if row['schools']>0:
            reasons.append("near schools")

        if row['colleges']>0:
            reasons.append("near colleges")

        if row['hospitals']>0:
            reasons.append("close to hospitals")

        if row['hotels']>0:
            reasons.append("near hotels")

        if row['competition']==0:
            reasons.append("very low competition")

        return ", ".join(reasons)


# ---------------------------------------------------
# SHOW TOP LOCATIONS
# ---------------------------------------------------

    st.subheader("Top 5 Suggested Locations")

    for i,row in best_locations.iterrows():

        reason = generate_reason(row)

        st.write(
            f"Location {best_locations.index.get_loc(i)+1} — "
            f"Lat: {row.lat:.5f}, Lon: {row.lon:.5f} — {reason}"
        )


# ---------------------------------------------------
# MAP
# ---------------------------------------------------

    m = folium.Map(
        location=[lat,lon],
        zoom_start=15,
        tiles="CartoDB positron"
    )


# competitors

    for _,row in competitors.iterrows():

        name = row['name'] if pd.notna(row['name']) else "Competitor"

        folium.Marker(
            [row.lat,row.lon],
            popup=name,
            icon=folium.Icon(color="red",icon="shopping-cart")
        ).add_to(m)


# infrastructure markers

    for _,row in schools.iterrows():

        folium.CircleMarker(
            [row.lat,row.lon],
            radius=4,
            color="blue",
            fill=True
        ).add_to(m)


# best locations

    for _,row in best_locations.iterrows():

        folium.CircleMarker(
            [row.lat,row.lon],
            radius=12,
            color="yellow",
            fill=True,
            fill_color="yellow",
            fill_opacity=0.9,
            popup="Suggested Location"
        ).add_to(m)


    st.subheader("Map")

    st_folium(m,width=1000,height=600)