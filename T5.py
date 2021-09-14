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

# df = pd.read_csv("musae_facebook_edges.csv")                            #load database from csv file

# g = nx.from_pandas_edgelist(df, source="id_1", target="id_2")      #graph information
# musae_facebook_target = pd.read_csv('musae_facebook_target.csv', usecols = ['id','page_type'] )

# df = pd.DataFrame(musae_facebook_target)

# politician=df.loc[(df['page_type'] =='company') ]

# gP=g.subgraph(list(politician['id'].values))
# tvshows=df.loc[(df['page_type'] =='government') ]
# gP_TV = g.subgraph(list(politician['id'].values)+list(tvshows['id'].values))

# gtvshows=g.subgraph(list(tvshows['id'].values))
# company=df.loc[(df['page_type'] =='company') ]
# gTV_company=g.subgraph(list(tvshows['id'].values)+list(company['id'].values)) 
# government=df.loc[(df['page_type'] =='government') ]
# gP_government = g.subgraph(list(politician['id'].values)+list(government['id'].values)) 

# pageType = {}
# color_map = []
# t = []
# t1 = []
# t2 = []
# for node in gP_TV.nodes:
#     if(node in gP.nodes ):
#         color_map.append('red') 
#         t1.append(node)
#         pageType[node]=1
#     else:
#         color_map.append('green')
#         t2.append(node) 
#         pageType[node]=2
# nx.set_node_attributes(gP_TV, pageType, "page_type")

# for node in gP_government.nodes:
#     if(node in gP.nodes ):
#         color_map.append('yellow') 
#         t1.append(node)
#         pageType[node]=1
#     else:
#         color_map.append('green')
#         t2.append(node) 
#         pageType[node]=2

# for node in gTV_company.nodes:
#     if(node in gtvshows.nodes ):
#         color_map.append('blue') 
#         t1.append(node)
#         pageType[node]=1
#     else:
#         color_map.append('red')
#         t2.append(node) 
#         pageType[node]=2

# nx.set_node_attributes(gP_TV, pageType, "page_type")
# # nx.set_node_attributes(gTV_company, pageType, "page_type")
# t = [t1, t2]
# print(nx.algorithms.community.modularity(gP_TV ,t,weight='None'))
# print(nx.numeric_assortativity_coefficient(gP_TV, "page_type"))
# print(nx.info(gP_TV))
# # amount of connected component
# cs = print( 'The number Connected Components in original graph: {0}'.format(nx.number_connected_components(gP_TV)))
# s = sorted(nx.connected_components(gP_TV), key=len, reverse=True)
# erGiant = gP_TV.subgraph(s[0])
# print(nx.info(erGiant))
# nx.draw(gP_TV, node_color=color_map, with_labels=False, font_size=8)
# plt.show()
H = nx.read_gml("adjnoun.gml")

print(H.nodes(data=True))
# nx.draw(H, with_labels=False, font_size=8)

print(nx.numeric_assortativity_coefficient(H, "value"))
#סכום דרגות 

# l=list(gP_TV.nodes.data("page_type"))
# print(degrees)
# print(l)
# for i in range(len(l)):
#     sum[l[i][1]-1]+=degrees[i][1]
# # print(sum)
# fig, ax = plt.subplots()
# deg=[1,2]
# colors=["yellow","blue"]

# plt.bar(deg, sum, width=0.50, color=colors,alpha=0.7)
# plt.legend(['politician','tvshow'])
# plt.title("Sum of Degrees")
# plt.ylabel("sum")
# # plt.xlabel("degree")
# plt.show()