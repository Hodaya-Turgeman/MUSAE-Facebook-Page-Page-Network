from networkx.classes.graph import Graph
import snap
import sparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import collections
from matplotlib import pylab

df = pd.read_csv("musae_facebook_edges.csv")                            #load database from csv file

g = nx.from_pandas_edgelist(df, source="id_1", target="id_2")      #graph information
musae_facebook_target = pd.read_csv('musae_facebook_target.csv', usecols = ['id','page_type'] )

df = pd.DataFrame(musae_facebook_target)
politician=df.loc[(df['page_type'] =='politician') ]
                                        #Adding adam to the list

nP = g.subgraph(politician['id'].values)  
government=df.loc[(df['page_type'] =='government') ]
nT = g.subgraph(list(politician['id'].values)+list(government['id'].values))  
nTv=g.subgraph(government['id'].values)
dg= nx.Graph()
dg.add_nodes_from(nT.nodes)
dg.add_edges_from(nT.edges-nP.edges-nTv.edges)

color_map = []
for node in dg.nodes:
    if(node in nTv.nodes ):
        color_map.append('pink') 
    else:
        color_map.append('blue') 
    
nx.draw(dg,node_color=color_map,node_size=15)
plt.savefig('file1.png')
# plt.show()
# nx.draw(nG, with_labels = True)    
# plt.show()  
# print(nx.info(g))

# print( 'The number Connected Components: {0}'.format(nx.number_connected_components(g))) 
# print('The diameter of graph is: {0}'.format(nx.diameter(g)))




