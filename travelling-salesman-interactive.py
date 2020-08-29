# distances and coordinates from https://www.distanciasentreciudades.com/
import streamlit as st
import pydeck as pdk
import pandas as pd
import json
import dimod
import networkx as nx
import dwave_networkx as dnx
import itertools
import neal
import numpy as np

st.sidebar.markdown('## Travelling salesman annealing')

@st.cache
def load_cities():
    data = pd.read_json('cities.json')
    return data

@st.cache
def load_distances():
    data = pd.read_json('distances.json')
    return data

cities = load_cities()
distances = load_distances()


selected_cities = st.sidebar.multiselect(
     'Select cities',
     options=list(cities['name']), 
     default=list(cities['name'])
     )

initial_city = st.sidebar.selectbox('Starting point',
selected_cities 
)



# filter dataframe
selectedCityData = cities[cities['name'].isin(selected_cities)]
# sort by id
selectedCityData.sort_values(by=['id'])


def getGraph(cityList,distances):    
    G = nx.Graph()
    distance_list= []
    for pair in itertools.combinations(cityList['id'], 2):
        dist=distances.loc[(distances['id1'] == pair[0]) & (distances['id2'] == pair[1])]
        distance_list.append([pair[0], pair[1], dist.iloc[0]['distance']])
    G.add_weighted_edges_from(distance_list)
    #st.write(G.edges.data())
    return G

citiesGraph=getGraph(selectedCityData,distances)
initial_city_code = cities.loc[cities.name == initial_city].iloc[0]['id']

solution=dnx.traveling_salesperson(citiesGraph, neal.SimulatedAnnealingSampler()) 

#rotate solution to set initial_city_code in first position
idx = solution.index(initial_city_code)
solution = solution[idx:] + solution[:idx]
#st.write(solution)

originCity = cities.loc[cities.id == solution[0]]
destCity = cities.loc[cities.id == solution[1]]
#sort to filter distances (id1<id2 in distances.json)
originCity_id=originCity.iloc[0]['id']
destCity_id=destCity.iloc[0]['id']
if (originCity_id>destCity_id):
    #swap ids
    originCity_id, destCity_id= destCity_id, originCity_id

distance_df=distances.loc[(distances['id1']==originCity_id) & (distances['id2']==destCity_id)]
step = {'step' : 1,
        'origin': originCity.iloc[0]['name'],
        'destination':destCity.iloc[0]['name'],
        'lat_ori': originCity.iloc[0]['lat'],
        'lon_ori': originCity.iloc[0]['lon'],
        'lat_dest':destCity.iloc[0]['lat'],
        'lon_dest':destCity.iloc[0]['lon'], 
        'distance': distance_df.iloc[0]['distance']
        }

#add first step
path_df = pd.DataFrame([], columns = ['step','origin','destination','lat_ori','lon_ori','lat_dest','lon_dest','distance'])
path_df = path_df.append(step,ignore_index=True)

#add the rest of steps to the path
for i in range(1,len(solution)-1):
    originCity = cities.loc[cities.id == solution[i]]
    destCity = cities.loc[cities.id == solution[i+1]]
    #sort to filter distances (id1<id2 in distances.json)
    originCity_id=originCity.iloc[0]['id']
    destCity_id=destCity.iloc[0]['id']
    if (originCity_id>destCity_id):
        #swap ids
        originCity_id, destCity_id= destCity_id, originCity_id

    distance_df=distances.loc[(distances['id1']==originCity_id) & (distances['id2']==destCity_id)]
    step = {'step' : i+1,
            'origin': originCity.iloc[0]['name'],
            'destination':destCity.iloc[0]['name'],
            'lat_ori': originCity.iloc[0]['lat'],
            'lon_ori': originCity.iloc[0]['lon'],
            'lat_dest':destCity.iloc[0]['lat'],
            'lon_dest':destCity.iloc[0]['lon'], 
            'distance': distance_df.iloc[0]['distance']
            }
    #append row to the dataframe
    path_df = path_df.append(step,ignore_index=True)


st.markdown('### Optimal path visiting selected cities')
st.write(path_df[['origin','destination','distance']])

GREEN_RGB = [0, 255, 0, 90]
RED_RGB = [240, 100, 0, 95]

# Specify a deck.gl ArcLayer
#https://deck.gl/docs/api-reference/layers/arc-layer
arc_layer = pdk.Layer(
    "ArcLayer",
    data=path_df,
    get_width=8,
    get_height=0,
    get_width_scale=6,
    get_source_position=["lon_ori","lat_ori"],
    get_target_position=["lon_dest","lat_dest"],
    get_tilt=5,
    get_source_color=RED_RGB,
    get_target_color=GREEN_RGB,
    pickable=True,
    auto_highlight=True,
)

#todo: TripsLayer?

cities_layer = pdk.Layer(
    "ScatterplotLayer",
    data=cities,
    get_position='[lon, lat]',
    get_color='[200, 30, 0]',
    get_radius=600,
    opacity=0.3
)

view_state = pdk.ViewState(
    longitude=2.09,
    latitude=41.32,
    zoom=10,
    min_zoom=5,
    max_zoom=15,
    pitch=30,
    bearing=-27.36)


st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=view_state,
    layers=[
        arc_layer,
        cities_layer 
    ],
    tooltip={
    "html": "<b>step:</b> {step}"
    "<br/> <b>from:</b> {origin}"
    " <br/> <b>to:</b> {destination} ",
    "style": {"color": "white"}
    },

))