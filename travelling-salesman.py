from __future__ import division
import itertools
from collections import defaultdict
from dwave_networkx.utils import binary_quadratic_model_sampler
import networkx as nx
import dwave_networkx as dnx
import neal

#Dictionary of real cities
cities_dict = {0:'El Prat de Llobregat', 1:'Cornellà de Llobregat',
 2:'Sant Feliu de Llobregat', 3:'Santa Coloma de Gramenet'}
num_cities = len(cities_dict)

#Set the problem as a graph:
#Solution: a sequence of ordered numbers, each corresponding to one city

#Create graph
graph = nx.Graph()
#Distances between cities taken from https://www.distanciasentreciudades.com/
#The weights of the edges of the graph correspond to the distances between cities
graph.add_weighted_edges_from({(0, 1, 3.2), (0, 2, 10.25),
(0, 3, 17.56), (1, 2, 3.69),(1, 3, 15.82), (2, 3, 14.2)})

#Choose starting point
initial_city = 'Cornellà de Llobregat'
for city_code in range(0,num_cities):
    if cities_dict[city_code] == initial_city:
        initial_city_code = city_code

#Use d-wave function to calculate the optimal route with a simulated annealing sampler
route = dnx.traveling_salesperson(graph, neal.SimulatedAnnealingSampler(),start=initial_city_code)
idx = route.index(initial_city_code)
route = route[idx:] + route[:idx]

#Get the names of the ordered cities in the optimal route
cities_route = []
for city_code in range(0,num_cities):
    cities_route.append(cities_dict[route[city_code]])

#Print the route, in this example it would print
#['Cornellà de Llobregat','El Prat de Llobregat', 'Sant Feliu de Llobregat', 'Santa Coloma de Gramenet']
print(cities_route)