import networkx as nx
import matplotlib.pyplot as plt
import community
import collections
import numpy as np
from scipy.stats import binom
import scipy.special
import csv

# ייבוא גרף
Data = open('Ass3_stormofswords.csv', "r")
next(Data, None)  # skip the first line in the input file
Graphtype = nx.Graph()
G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype, nodetype=str, data=(('Weight', int),))
original_nodes = list(G.nodes)
original_edges = list(G.edges)
print(nx.info(G))
num_of_nodes = len(original_nodes)

# nx.draw(G, with_labels=False)
# plt.show()

tribes = {}
t = []
partition={}
with open('Ass3_tribes.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        tribes[row['node']] = int(row['tribe'])
        # x=row['node']
        # y=int(row['tribe'])
        t.append(int(row['tribe']))
print(tribes)
nx.set_node_attributes(G, tribes, "tribe")

color_map = []
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
t6 = []
t7 = []
for k, v in nx.get_node_attributes(G,'tribe').items():
    if v == 1:
        color_map.append('blue')
        t1.append(k)
    elif v == 2:
        color_map.append('green')
        t2.append(k)
    elif v == 3:
        color_map.append('red')
        t3.append(k)
    elif v == 4:
        color_map.append('yellow')
        t4.append(k)
    elif v == 5:
        color_map.append('pink')
        t5.append(k)
    elif v == 6:
        color_map.append('orange')
        t6.append(k)
    elif v == 7:
        color_map.append('purple')
        t7.append(k)
# nx.draw(G, node_color=color_map, with_labels=False, font_size=8)
# plt.show()

# t = [t1, t2, t3, t4, t5, t6, t7]

#ציור לפי גדלי צמתים
# degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
# node_sizes=[]
# for x in degree_sequence:
#     node_sizes.append(x*50)
# nx.draw(G, node_color=color_map,node_size=node_sizes, with_labels=False,alpha=0.8)
# plt.show()

# ציור עם משקלים
# pos=nx.spring_layout(G)
# nx.draw(G,pos,node_color=color_map,with_labels=False,node_size=node_sizes,alpha=0.8)
# labels = nx.get_edge_attributes(G,'Weight')
# nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_size=7,font_color='black')
# plt.show()

# print(nx.algorithms.community.modularity(G, t,weight='Weight'))
# print(nx.numeric_assortativity_coefficient(G, "tribe"))

#סכום דרגות 
# sum=[0,0,0,0,0,0,0]
# degrees = list(G.degree())
# l=list(G.nodes.data("tribe"))
# print(degrees)
# print(l)
# for i in range(len(l)):
#     sum[l[i][1]-1]+=degrees[i][1]
# print(sum)
# fig, ax = plt.subplots()
# deg=[1,2,3,4,5,6,7]
# colors=["blue","green","red","yellow","pink","orange","purple"]
# plt.bar(deg, sum, width=0.80, color=colors)
# plt.title("Sum of Degrees")
# plt.ylabel("sum")
# plt.xlabel("tribe")
# plt.show()
