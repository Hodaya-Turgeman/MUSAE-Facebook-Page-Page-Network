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
gP = g.subgraph(politician['id'].values)  
tvshows=df.loc[(df['page_type'] =='tvshow') ]
gTv=g.subgraph(tvshows['id'].values)
government=df.loc[(df['page_type'] =='government') ]
gGoverment=g.subgraph(government['id'].values)
company=df.loc[(df['page_type'] =='company') ]
gC=g.subgraph(company['id'].values)
cs = print( 'The number Connected Components in politician graph: {0}'.format(nx.number_connected_components(gP)))
# print("politician",nx.info(gP))
cs = print( 'The number Connected Components in tvshow graph: {0}'.format(nx.number_connected_components(gTv)))
# print('tvshow',nx.info(gTv))
print( 'The number Connected Components in goverment graph: {0}'.format(nx.number_connected_components(gGoverment)))
# print('government',nx.info(gGoverment))
print( 'The number Connected Components in company graph: {0}'.format(nx.number_connected_components(gC)))
# print("company",nx.info(gC))

degree_sequenceP = sorted([f for m, f in gP.degree()], reverse=True)  # degree sequence of gilbert
degree_sequenceTv = sorted([f for m, f in gTv.degree()], reverse=True)  # degree sequence of gilbert
degree_sequenceG = sorted([f for m, f in gGoverment.degree()], reverse=True)  # degree sequence of gilbert
degree_sequenceC = sorted([f for m, f in gC.degree()], reverse=True)  # degree sequence of gilbert
in_histP = range(len(degree_sequenceP))
in_histTv = range(len(degree_sequenceTv))
in_histG = range(len(degree_sequenceG))
in_histC = range(len(degree_sequenceC))
plt.figure() # you need to first do 'import pylab as plt'
plt.grid(True)
plt.loglog(in_histP, degree_sequenceP, 'yo-') # source graph degree
plt.loglog(in_histTv, degree_sequenceTv, 'bv-') # gilbert graph degree
plt.loglog(in_histG, degree_sequenceG, 'gx-') # gilbert graph degree
plt.loglog(in_histC, degree_sequenceC, 'ro-') # gilbert graph degree
plt.legend(['Politician', 'Tvshow','Government','Company'])
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Facebook')
# plt.xlim([0, 1.15*10**2])
plt.show()
