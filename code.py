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

df = pd.read_csv("musae_facebook_edges.csv")                            #load database from csv file

g = nx.from_pandas_edgelist(df, source="id_1", target="id_2")      #graph information
musae_facebook_target = pd.read_csv('musae_facebook_target.csv', usecols = ['id','page_type'] )

# df = pd.DataFrame(musae_facebook_target)
# politician=df.loc[(df['page_type'] =='politician') ]
                                         #Adding adam to the list

# nP = g.subgraph(politician['id'].values)  
# tvshows=df.loc[(df['page_type'] =='tvshow') ]
# nT = g.subgraph(list(politician['id'].values)+list(tvshows['id'].values))  
# nTv=g.subgraph(tvshows['id'].values)
# dg= nx.Graph()
# dg.add_nodes_from(nT.nodes)
# dg.add_edges_from(nT.edges-nP.edges-nTv.edges)

# color_map = []
# for node in dg.nodes:
#     if(node in nTv.nodes ):
#         color_map.append('pink') 
#     else:
#         color_map.append('blue') 

# degrees = [dg.degree(n) for n in dg.nodes()]  #A list with all the graph's degrees
# num0 = degrees.count(0)
# num1 = degrees.count(1)
# print('num0: ', num0)
# print('num 1: ', num1)

# cen = nx.closeness_centrality(g)

# res = dict(sorted(cen.items(), key = itemgetter(1), reverse = True)[:5])
# print("The top N value pairs are  " + str(res))

# cenval = cen.values()
# min_value = min(cenval)
# min_key = min(cen, key=cen.get)
# print(min_key , min_value)

# newA = dict(sorted(cen.iteritems(), key=cen.values(), reverse=True)[:5])
# print(newA)
# avg = sum(cen.values()) / float(len(cen))
# print('closnes average', avg)



# bit = nx.betweenness_centrality(g)
# resb = dict(sorted(bit.items(), key = itemgetter(1), reverse = True)[:5])
# print("The top 5 value pairs are  " + str(resb))

# bitval = bit.values()
# min_value = min(bitval)
# min_key = min(bit, key=bit.get)
# print(min_key , min_value)


# avgb = sum(bit.values()) / float(len(bit))
# print('betweeness average', avgb)

deg = nx.degree_centrality(g)
resd = dict(sorted(deg.items(), key = itemgetter(1), reverse = True)[:5])
print("The top 5 value pairs are  " + str(resd))

degval = deg.values()
min_value = min(degval)
min_key = min(deg, key=deg.get)
print(min_key , min_value)
# avgd = sum(deg.values()) / float(len(deg))
# print('degree average', avgd)
# print(nx.diameter(g))
# print('diameter')
#print(nx.info(dg))
#print( 'The number Connected Components: {0}'.format(nx.number_connected_components(dg)))      #Print the number of connected components in the nieghbors graph



# nx.draw_random(dg,node_color=color_map,node_size=15)
# nx.draw(dg,node_color=color_map,node_size=15)
#plt.savefig('file1.png')
# plt.show()


# nx.draw(nG, with_labels = True)    
# plt.show()  
# print(nx.info(g))

# print( 'The number Connected Components: {0}'.format(nx.number_connected_components(g))) 
# print('The diameter of graph is: {0}'.format(nx.diameter(g)))


#degree centrality


# closeness1=nx.closeness_centrality(dg)
# node_sizes=[]
# for x in closeness1.values():
#      node_sizes.append(x*100000)


# nx.draw(dg, node_color=color_map,node_size=node_sizes, with_labels=False)
# # # # pos = nx.spring_layout(G)
# plt.show()

# betweenness1=nx.betweenness_centrality(dg)
# node_sizes=[]
# for x in betweenness1.values():
#      node_sizes.append(x*1000000)


# nx.draw(dg, node_color=color_map,node_size=node_sizes, with_labels=False)
# # # # pos = nx.spring_layout(G)
# plt.show()

# degcen1=nx.degree_centrality(dg)
# node_sizes=[]
# for x in degcen1.values():
#      node_sizes.append(x*100000)


# nx.draw(dg, node_color=color_map,node_size=node_sizes, with_labels=False)
# # # # pos = nx.spring_layout(G)
# plt.show()