from networkx.classes.graph import Graph
import snap
import sparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import collections
from matplotlib import pylab
from operator import itemgetter
from collections import defaultdict
from scipy.cluster import hierarchy
from scipy.spatial import distance
import community as community_louvain
import seaborn as sns
from numpy import random
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
import networkx.algorithms.community

# G = nx.Graph()
# df = pd.read_csv("Ass3_stormofswords.csv")                            #load database from csv file

G = pd.read_csv('Ass3_stormofswords.csv', usecols = ['Source','Target', 'Weight'] )

G = nx.from_pandas_edgelist(G, source = 'Source', target = 'Target' )
# G = nx.read_weighted_edgelist('Ass3_stormofswords.csv')
# P = pd.read_csv('Ass3_tribes.csv')
# P = pd.DataFrame(P)

# P = P.set_index('id_1').T.to_dict('list')


d = dict()
f = open("Ass3_tribes.csv")
for line in f:
    line = line.strip('\n')
    (key, val) = line.split(",")
    d[key] = val

print(d)


# for i in G.nodes:
#     G.nodes[i]['tribe'] = P[i][0]


nx.set_node_attributes(G, d, "tribe")
# print(G.nodes)
# tribes = [[] for x in range(7)]  
# tribe0=P.loc[(P['id_2'] ==1)]
# tribes[0] = set(tribe0['id_1'].values)
# tribe1=P.loc[(P['id_2'] ==2)]
# tribes[1] = set(tribe0['id_1'].values)
# tribe2=P.loc[(P['id_2'] ==3)]
# tribes[2] = set(tribe0['id_1'].values)
# tribe3=P.loc[(P['id_2'] ==4)]
# tribes[3] = set(tribe0['id_1'].values)
# tribe4=P.loc[(P['id_2'] ==5)]
# tribes[4] = set(tribe0['id_1'].values)
# tribe5=P.loc[(P['id_2'] ==6)]
# tribes[5] = set(tribe0['id_1'].values)
# tribe6=P.loc[(P['id_2'] ==7)]
# tribes[6] = set(tribe0['id_1'].values)
# print(type(tribes))


# print(nx.algorithms.community.modularity(G))


# part = community_louvain.best_partition(G, partition=d, weight='Weight', resolution=1.0, randomize=None, random_state=None)
mod = community_louvain.modularity(d, G, weight='Weight')

# Plot, color nodes using community structure
# values = [part.get(node) for node in G.nodes()]
values = [d.get(node) for node in G.nodes]
color_map = []
for node in G.nodes:
    if d.get(node) == '1':
        color_map.append('blue')
    if d.get(node) == '2':
        color_map.append('yellow')
    if d.get(node) == '3':
        color_map.append('pink')
    if d.get(node) == '4':
        color_map.append('red')
    if d.get(node) == '5':
        color_map.append('purple')
    if d.get(node) == '6':
        color_map.append('brown')
    if d.get(node) == '7': 
        color_map.append('green') 


# nx.draw_spring(G, cmap=plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)
nx.draw(G, node_color = color_map, node_size=30, with_labels=False)
plt.show()
# print('pppppp',part)
print(mod)