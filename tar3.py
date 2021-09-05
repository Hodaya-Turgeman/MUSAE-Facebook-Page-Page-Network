
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

#load database from csv file
dataf = pd.read_csv("Ass3_stormofswords.csv")      
gg = nx.from_pandas_edgelist(dataf, source="Source", target="Target")      #graph information
stormofswords_target = pd.read_csv('Ass3_stormofswords.csv', usecols = ['Source','Target','Weight'] )
#source graph information
# nx.draw_random(gg,node_size=30)
# plt.show()
# print('######Source#####')
# print(nx.info(gg))


#erdos renyi random graph model
erg = nx.erdos_renyi_graph(107, 0.06189)    
# nx.draw_random(erg,node_size=30)
# plt.show()
# print('######Erdos Renyi#####')
# print(nx.info(erg))


#gilbert random graph model
grg = nx.gnm_random_graph(107, 353)          
# nx.draw_random(grg,node_size=30)
# plt.show()
# print('######Gilbert#####')
# print(nx.info(grg)) 


#configuration model
degree_sequence = [d for n, d in gg.degree()]  
cmg=nx.configuration_model(degree_sequence)
# nx.draw_random(grg,node_size=30)
# plt.show()
# print('######Configuration Model#####')
# print(nx.info(cmg)) 

#degree distribution original vs. erdosh reyni
degree_sequenceS = sorted([d for n, d in gg.degree()], reverse=True)  # degree sequence of source
degree_sequenceE = sorted([d for n, d in erg.degree()], reverse=True)  # degree sequence of erdosh
in_histS = range(len(degree_sequenceS))
in_histE = range(len(degree_sequenceE))
plt.figure() # you need to first do 'import pylab as plt'
plt.grid(True)
plt.loglog(in_histS, degree_sequenceS, 'ro-') # source graph degree
plt.loglog(in_histE, degree_sequenceE, 'bv-') # erdosh graph degree
plt.legend(['Source graph', 'Erdosh Reyni graph'])
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Games of Thrones')
plt.xlim([0, 1.15*10**2])
plt.show()


degreeCount = collections.Counter(degree_sequenceS)
degreeCount1 = collections.Counter(degree_sequenceE)
deg, cnt = zip(*degreeCount.items())
deg1, cnt1 = zip(*degreeCount1.items())
fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color="r")
plt.bar(deg1, cnt1, width=0.80, color="b")
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.xscale('log')
plt.show()


#degree distribution original vs. gilbert
degree_sequenceG = sorted([f for m, f in grg.degree()], reverse=True)  # degree sequence of gilbert
in_histG = range(len(degree_sequenceG))
plt.figure() # you need to first do 'import pylab as plt'
plt.grid(True)
plt.loglog(in_histS, degree_sequenceS, 'ro-') # source graph degree
plt.loglog(in_histG, degree_sequenceG, 'gv-') # gilbert graph degree
plt.legend(['Source graph', 'Gilbert graph'])
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Games of Thrones')
plt.xlim([0, 1.15*10**2])
plt.show()


degreeCount2 = collections.Counter(degree_sequenceG)
deg2, cnt2 = zip(*degreeCount2.items())
fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color="r")
plt.bar(deg2, cnt2, width=0.80, color="g")
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.xscale('log')
plt.show()

# degree distribution original vs. congiguration
degree_sequenceC = sorted([f for m, f in cmg.degree()], reverse=True)  # degree sequence of configuration model
in_histC = range(len(degree_sequenceC))
plt.figure() # you need to first do 'import pylab as plt'
plt.grid(True)
plt.loglog(in_histS, degree_sequenceS, 'ro-') # source graph degree
plt.loglog(in_histC, degree_sequenceC, 'yx-') # configuratioon graph degree
plt.legend(['Source graph', 'Configuration graph'])
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Games of Thrones')
plt.xlim([0, 1.15*10**2])
plt.show()

degreeCount3 = collections.Counter(degree_sequenceC)
deg3, cnt3 = zip(*degreeCount3.items())
fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color="r")
plt.bar(deg3, cnt3, width=0.80, color="y")
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.xscale('log')
plt.show()


#degree distribution erdosh reyni vs. configuration vs. gilbert
plt.figure() # you need to first do 'import pylab as plt'
plt.grid(True)
plt.loglog(in_histE, degree_sequenceE, 'bv-') # erdosh graph degree
plt.loglog(in_histG, degree_sequenceG, 'go-') # gilbert graph degree
plt.loglog(in_histC, degree_sequenceC, 'yx-') # configuratioon graph degree
plt.legend(['Erdosh Reyni graph', 'Gilbert graph', 'Configuration graph'])
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Games of Thrones')
plt.xlim([0, 1.15*10**2])
plt.show()


fig, ax = plt.subplots()
plt.bar(deg1, cnt1, width=0.80, color="b")
plt.bar(deg2, cnt2, width=0.80, color="g")
plt.bar(deg3, cnt3, width=0.80, color="y")
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.xscale('log')
plt.show()

#amount of connected component
# cs = print( 'The number Connected Components in original graph: {0}'.format(nx.number_connected_components(gg))) 
# ce = print( 'The number Connected Components in erdosh reyni graph: {0}'.format(nx.number_connected_components(erg))) 
# cg = print( 'The number Connected Components in gilbert graph: {0}'.format(nx.number_connected_components(grg))) 
# cc = print( 'The number Connected Components in configuration graph: {0}'.format(nx.number_connected_components(cmg))) 

#gaint component of erdosh reyni 
s = sorted(nx.connected_components(erg), key=len, reverse=True)
erGiant = erg.subgraph(s[0])
# nodes=list(erGiant.nodes)
# edges=list(erGiant.edges)
# print("largest connected components nodes",nodes)
# print("largest connected components edges",edges)
# print("num of nodes and edges", len(nodes),len(edges))
# nx.draw(erGiant,pos=nx.circular_layout(erg),with_labels=False)
# plt.show()

#gaint component of gilbert
g = sorted(nx.connected_components(grg), key=len, reverse=True)
grGiant = grg.subgraph(g[0])
# nodes=list(grGiant.nodes)
# edges=list(grGiant.edges)
# print("largest connected components nodes",nodes)
# print("largest connected components edges",edges)
# print("num of nodes and edges", len(nodes),len(edges))
# nx.draw(grGiant,pos=nx.circular_layout(grg),with_labels=False)
# plt.show()

#gaint component of configuration model
g = sorted(nx.connected_components(cmg), key=len, reverse=True)
cmGiant = cmg.subgraph(g[0])
# nodes=list(cmGiant.nodes)
# edges=list(cmGiant.edges)
# print("largest connected components nodes",nodes)
# print("largest connected components edges",edges)
# print("num of nodes and edges", len(nodes),len(edges))
# nx.draw(cmGiant,pos=nx.circular_layout(cmg),with_labels=False)
# plt.show()


#Average shortest path
# print( 'The average shortest path length in original graph: {0}'.format(nx.average_shortest_path_length(gg)))
# if ce == 1:
#     print( 'The average shortest path length in erdosh reyni graph: {0}'.format(nx.average_shortest_path_length(erg)))
# else:
#     print( 'The average shortest path length of the giant component in erdosh reyni graph: {0}'.format(nx.average_shortest_path_length(erGiant)))
# if cg == 1:    
#     print( 'The average shortest path length in gilbert graph: {0}'.format(nx.average_shortest_path_length(grg)))
# else:
#     print( 'The average shortest path length of the giant component in gilbert graph: {0}'.format(nx.average_shortest_path_length(grGiant)))
# if cc == 1:    
#     print( 'The average shortest path length in configuration graph: {0}'.format(nx.average_shortest_path_length(cmg)))
# else:
#     print( 'The average shortest path length of the giant component in configuration graph: {0}'.format(nx.average_shortest_path_length(cmGiant)))








#our Facebook graph
#load graph from csv file
# dfile = pd.read_csv("musae_facebook_edges.csv")                        #load database from csv file

# facebook1 = nx.from_pandas_edgelist(dfile, source="id_1", target="id_2")      #graph information
# musae_facebook_target = pd.read_csv('musae_facebook_target.csv', usecols = ['id','facebook_id', 'page_name', 'page_type'] )
# df = pd.DataFrame(musae_facebook_target)       
# # print(nx.info(facebook1))


# #erdos renyi random graph model
# Serg = nx.erdos_renyi_graph(22470, 0.00068)    
# # nx.draw_random(erg,node_size=30)
# # plt.show()
# # print('######Facebook Erdos Renyi#####')
# # print(nx.info(Serg))


# #gilbert random graph model
# Sgrg = nx.gnm_random_graph(22470, 171002)          
# # nx.draw_random(grg,node_size=30)
# # plt.show()
# # print('######Facebook Gilbert#####')
# # print(nx.info(Sgrg)) 


# #configuration model
# degree_sequenceF = [d for n, d in facebook1.degree()]  
# Scmg=nx.configuration_model(degree_sequenceF)
# # nx.draw_random(grg,node_size=30)
# # plt.show()
# # print('######Facebook Configuration Model#####')
# # print(nx.info(Scmg)) 

# # degree distribution original vs. erdosh reyni
# degree_sequenceSF = sorted([d for n, d in facebook1.degree()], reverse=True)  # degree sequence of source
# degree_sequenceEF = sorted([d for n, d in Serg.degree()], reverse=True)  # degree sequence of erdosh
# in_histSF = range(len(degree_sequenceSF))
# in_histEF = range(len(degree_sequenceEF))
# plt.figure() # you need to first do 'import pylab as plt'
# plt.grid(True)
# plt.loglog(in_histSF, degree_sequenceSF, 'ro-') # source graph degree
# plt.loglog(in_histEF, degree_sequenceEF, 'bv-') # erdosh graph degree
# plt.legend(['Source graph', 'Erdosh Reyni graph'])
# plt.xlabel('Degree')
# plt.ylabel('Number of nodes')
# plt.title('Facebook')
# plt.xlim([0, 2*10**5])
# plt.ylim([0, 2*10**3])
# plt.show()

# degreeCount1 = collections.Counter(degree_sequenceSF)
# degreeCount = collections.Counter(degree_sequenceEF)
# deg, cnt = zip(*degreeCount.items())
# deg1, cnt1 = zip(*degreeCount1.items())
# fig, ax = plt.subplots()
# plt.bar(deg, cnt, width=0.80, color="b")
# plt.bar(deg1, cnt1, width=0.80, color="r")
# plt.xlabel('Degree')
# plt.ylabel('Number of nodes')
# plt.xscale('log')
# plt.show()


# # degree distribution original vs. gilbert
# degree_sequenceGF = sorted([d for n, d in Sgrg.degree()], reverse=True)  # degree sequence of erdosh
# in_histGF = range(len(degree_sequenceGF))
# plt.figure() # you need to first do 'import pylab as plt'
# plt.grid(True)
# plt.loglog(in_histSF, degree_sequenceSF, 'ro-') # source graph degree
# plt.loglog(in_histGF, degree_sequenceGF, 'gv-') # gilbert graph degree
# plt.legend(['Source graph', 'Gilbert graph'])
# plt.xlabel('Degree')
# plt.ylabel('Number of nodes')
# plt.title('Facebook')
# plt.xlim([0, 10**5])
# plt.ylim([0, 10**3])
# plt.show()


# degreeCount2 = collections.Counter(degree_sequenceGF)
# deg2, cnt2 = zip(*degreeCount2.items())
# fig, ax = plt.subplots()
# plt.bar(deg2, cnt2, width=0.80, color="g")
# plt.bar(deg1, cnt1, width=0.80, color="r")
# plt.xlabel('Degree')
# plt.ylabel('Number of nodes')
# plt.xscale('log')
# plt.show()



# # degree distribution original vs. configuration
# degree_sequenceCF = sorted([d for n, d in Scmg.degree()], reverse=True)  # degree sequence of erdosh
# in_histCF = range(len(degree_sequenceCF))
# plt.figure() # you need to first do 'import pylab as plt'
# plt.grid(True)
# plt.loglog(in_histSF, degree_sequenceSF, 'ro-') # source graph degree
# plt.loglog(in_histCF, degree_sequenceCF, 'yx-') # erdosh graph degree
# plt.legend(['Source graph', 'Configuration graph'])
# plt.xlabel('Degree')
# plt.ylabel('Number of nodes')
# plt.title('Facebook')
# plt.xlim([0, 10**5])
# plt.ylim([0, 10**3])
# plt.show()

# degreeCount3 = collections.Counter(degree_sequenceCF)
# deg3, cnt3 = zip(*degreeCount3.items())
# fig, ax = plt.subplots()
# plt.bar(deg3, cnt3, width=0.80, color="y")
# plt.bar(deg1, cnt1, width=0.80, color="r")
# plt.xlabel('Degree')
# plt.ylabel('Number of nodes')
# plt.xscale('log')
# plt.show()



# # degree distribution erdosh vs. gilbert vs. configuration
# plt.figure() # you need to first do 'import pylab as plt'
# plt.grid(True)
# plt.loglog(in_histEF, degree_sequenceEF, 'bv-') # erdosh graph degree
# plt.loglog(in_histGF, degree_sequenceGF, 'go-') # gilbert
# plt.loglog(in_histCF, degree_sequenceCF, 'yx-') # erdosh graph degree
# plt.legend(['Erdosh Reyni graph', 'Gilbert graph', 'Configuration graph'])
# plt.xlabel('Degree')
# plt.ylabel('Number of nodes')
# plt.title('Facebook')
# plt.xlim([0, 10**5])
# plt.ylim([0, 10**3])
# plt.show()



# fig, ax = plt.subplots()
# plt.bar(deg2, cnt2, width=0.80, color="g")
# plt.bar(deg3, cnt3, width=0.80, color="y")
# plt.bar(deg, cnt, width=0.80, color="b")
# plt.xlabel('Degree')
# plt.ylabel('Number of nodes')
# plt.xscale('log')
# plt.show()

# amount of connected component
# Fcs = print( 'The number Connected Components in Facebook original graph: {0}'.format(nx.number_connected_components(facebook1))) 
# Fce = print( 'The number Connected Components in Facebook erdosh reyni graph: {0}'.format(nx.number_connected_components(Serg))) 
# Fcg = print( 'The number Connected Components in Facebook gilbert graph: {0}'.format(nx.number_connected_components(Sgrg))) 
# Fcc = print( 'The number Connected Components in Facebook configuration graph: {0}'.format(nx.number_connected_components(Scmg))) 

# gaint component of erdosh reyni 
# sF = sorted(nx.connected_components(Serg), key=len, reverse=True)
# erGiantF = Serg.subgraph(sF[0])
# nodes=list(erGiantF.nodes)
# edges=list(erGiantF.edges)
# print("largest connected components nodes",nodes)
# print("largest connected components edges",edges)
# print("num of nodes and edges", len(nodes),len(edges))
# nx.draw(erGiantF,pos=nx.circular_layout(Serg),with_labels=False)
# plt.show()
# print('######Facebook Erdosh Reyni Model giant component#####')
# print(nx.info(erGiantF))

# gaint component of gilbert
# gF = sorted(nx.connected_components(Sgrg), key=len, reverse=True)
# grGiantF = Sgrg.subgraph(gF[0])
# nodes=list(grGiantF.nodes)
# edges=list(grGiantF.edges)
# print("largest connected components nodes",nodes)
# print("largest connected components edges",edges)
# print("num of nodes and edges", len(nodes),len(edges))
# nx.draw(grGiantF,pos=nx.circular_layout(Sgrg),with_labels=False)
# plt.show()
# print('######Facebook Gilbert Model giant component#####')
# print(nx.info(grGiantF))


# gaint component of configuration model
# cF = sorted(nx.connected_components(Scmg), key=len, reverse=True)
# cmGiantF = Scmg.subgraph(cF[0])
# nodes=list(cmGiantF.nodes)
# edges=list(cmGiantF.edges)
# print("largest connected components nodes",nodes)
# print("largest connected components edges",edges)
# print("num of nodes and edges", len(nodes),len(edges))
# nx.draw(cmGiantF,pos=nx.circular_layout(Scmg),with_labels=False)
# plt.show()
# print('######Facebook Configuration Model giant component#####')
# print(nx.info(cmGiantF))

# Average shortest path
# print( 'The average shortest path length in original graph: {0}'.format(nx.average_shortest_path_length(facebook1)))
# if Fcs == 1:
#     print( 'The average shortest path length in erdosh reyni graph: {0}'.format(nx.average_shortest_path_length(Serg)))
# else:
#     print( 'The average shortest path length of the giant component in erdosh reyni graph: {0}'.format(nx.average_shortest_path_length(erGiantF)))
# if Fcg == 1:    
#     print( 'The average shortest path length in gilbert graph: {0}'.format(nx.average_shortest_path_length(Sgrg)))
# else:
#     print( 'The average shortest path length of the giant component in gilbert graph: {0}'.format(nx.average_shortest_path_length(grGiantF)))
# if Fcc == 1:    
#     print( 'The average shortest path length in configuration graph: {0}'.format(nx.average_shortest_path_length(Scmg)))
# else:
#     print( 'The average shortest path length of the giant component in configuration graph: {0}'.format(nx.average_shortest_path_length(cmGiantF)))