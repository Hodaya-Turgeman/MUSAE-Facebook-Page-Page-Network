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

df = pd.DataFrame(musae_facebook_target)

politician=df.loc[(df['page_type'] =='politician') ]

gP=g.subgraph(list(politician['id'].values))
tvshows=df.loc[(df['page_type'] =='tvshow') ]
government=df.loc[(df['page_type'] =='government') ]
company=df.loc[(df['page_type'] =='company') ]

g_tv_g_c=g.subgraph(list(government['id'].values)+list(tvshows['id'].values)+list(company['id'].values))
pageType = {}
for node in g.nodes:
    if(node in gP.nodes ):
        pageType[node]=1
    else:
        color_map.append('blue')
        # t2.append(node) 
        pageType[node]=2
nx.set_node_attributes(gP_TV, pageType, "page_type")

max_grade = max(gP.degree, key=lambda x: x[1])[1]
print(max_grade)
Adam = [node[0] for node in gP.degree if node[1] == max_grade]    #Node with the largest degree
Adam = Adam[0]
print("The strong in politication")
print("I am the node with the largest degree: ",Adam)
# List1 = list(g.neighbors(Adam))
# List2=list(gP.neighbors(Adam))                              #List of Adam's neighbors
# neighbors = [List1[i]-List2[i] for i in range(min(len(List1), len(List2)))]

# neighbors.append(Adam)                                              #Adding adam to the list

# gAdam = g.subgraph(neighbors) 

# print(nx.info(gAdam))

#   גרף המכיל את הצלעות שבין הפוליטיקים לשאר הקבוצות אבל אין צלעות בים פוליטיקאים לפוליטקאים או בין הקבוצות עצמם -מציאת קודקוד של הפוליטיקאי בעל קשר הכי גדול אם שאר הקבוצות 
l=list(g_tv_g_c.nodes)
l.append(Adam)
gP_and_Other= nx.Graph()
gt=g.subgraph(l)
gP_and_Other.add_nodes_from(g.nodes)
gP_and_Other.add_edges_from(g.edges-g_tv_g_c.edges-gP.edges)

print(g)
degree_sequenceF = sorted([d for n, d in gP_and_Other.degree()], reverse=True)  # degree sequence of source
for i in degree_sequenceF:
    B = [node[0] for node in gP_and_Other.degree if node[1] == i]    #Node with the largest degree
    B = B[0]
    if gP_and_Other.nodes[str(B)]['page_type']=='politician':
        break

print(nx.info(gP_and_Other))
print("I am the node with the largest degree  in gP_and_Other: ",B)
# max_grade = max(gP_and_Other.degree, key=lambda x: x[1])[1]
# print(max_grade)
# B = [node[0] for node in gP_and_Other.degree if node[1] == max_grade]    #Node with the largest degree
# B = B[0]
# print(nx.info(gP_and_Other))
# print("I am the node with the largest degree  in gP_and_Other: ",B)

# cen = nx.closeness_centrality(g)
# res = dict(sorted(cen.items(), key = itemgetter(1), reverse = True)[:5])
# print("The top N value pairs are  " + str(res))

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
#         color_map.append('yellow') 
#         # t1.append(node)
#         pageType[node]=1
#     else:
#         color_map.append('blue')
#         # t2.append(node) 
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
# nx.set_node_attributes(gTV_company, pageType, "page_type")
# t = [t1, t2]

# amount of connected component
# cs = print( 'The number Connected Components in original graph: {0}'.format(nx.number_connected_components(gP_TV)))
# s = sorted(nx.connected_components(gP_TV), key=len, reverse=True)
# erGiant = gP_TV.subgraph(s[0])
# print(nx.info(gP_TV))
# print("big####")
# print(nx.info(erGiant))


# t = []
# t1 = []
# t2 = []   
# for node in erGiant.nodes:
#     if(node in gP.nodes ):
#         color_map.append('red') 
#         pageType[node]=1
#         t1.append(node)
#     else:
#         color_map.append('green')
#         t2.append(node) 
#         pageType[node]=2
# nx.set_node_attributes(erGiant, pageType, "page_type")
# print("t1",len(t1))
# t = [t1, t2] 
# print(nx.algorithms.community.modularity(erGiant ,t,weight='None'))
# print(nx.numeric_assortativity_coefficient(erGiant, "page_type"))

# nx.draw(erGiant, node_color=color_map, with_labels=False, font_size=8)
# plt.show()
# H = nx.read_gml("adjnoun.gml")

# print(H.nodes(data=True))
# nx.draw(H, with_labels=False, font_size=8)

# print(nx.numeric_assortativity_coefficient(H, "value"))
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