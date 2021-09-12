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

gP_TV = g.subgraph(list(politician['id'].values)+list(tvshows['id'].values)) 
pageType = {}
# t = []
# with open('musae_facebook_target.csv', newline='') as csvfile:
#     reader =csv.DictReader(csvfile)
#     for row in reader:
#         target[row['node']] = int(row['page_type'])
#         # x=row['node']
#         # y=int(row['tribe'])
#         t.append(int(row['page_type']))
# nx.set_node_attributes(gP_TV, target, "tribe")
color_map = []
t = []
t1 = []
t2 = []
for node in gP_TV.nodes:
    if(node in gP.nodes ):
        color_map.append('yellow') 
        t1.append(node)
        pageType[node]=1
    else:
        color_map.append('blue')
        t2.append(node) 
        pageType[node]=2
nx.set_node_attributes(gP_TV, pageType, "page_type")
t = [t1, t2]
print(nx.algorithms.community.modularity(gP_TV, t,weight='None'))
print(nx.numeric_assortativity_coefficient(gP_TV, "page_type"))

# nx.draw(gP_TV, node_color=color_map, with_labels=False, font_size=8)
# plt.show()

#סכום דרגות 
sum=[0,0]
degrees = list(gP_TV.degree())
degree_sequenceC = sorted([f for m, f in gP_TV.degree()], reverse=True)  # degree sequence of configuration model
maxDeg = degree_sequenceC[0]
print(maxDeg)

l=list(gP_TV.nodes.data("page_type"))
print(degrees)
# print(l)
for i in range(len(l)):
    sum[l[i][1]-1]+=degrees[i][1]
# print(sum)
fig, ax = plt.subplots()
deg=[1,2]
colors=["yellow","blue"]

plt.bar(deg, sum, width=0.50, color=colors,alpha=0.7)
plt.legend(['politician','tvshow'])
plt.title("Sum of Degrees")
plt.ylabel("sum")
# plt.xlabel("degree")
plt.show()