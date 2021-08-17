from igraph import *

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections

# musae_facebook_target=pd.read_csv('musae_facebook_target.csv')
# musae_facebook_target['page_type']=musae_facebook_target['page_type'].str.replace('company','0',True)
# musae_facebook_target['page_type']=musae_facebook_target['page_type'].str.replace('government','1',True)
# musae_facebook_target['page_type']=musae_facebook_target['page_type'].str.replace('politician','2',True)
# musae_facebook_target['page_type']=musae_facebook_target['page_type'].str.replace('tvshow','3',True)

# df = pd.DataFrame(musae_facebook_target)
# df.to_csv('musae_facebook_target.csv')

n_vertices = 22470

# Create graph
g = Graph()

# Add vertices
g.add_vertices(n_vertices)
edges = []

df = pd.read_csv("musae_facebook_edges.csv")                            #load database from csv file

gn = nx.from_pandas_edgelist(df, source="id_1", target="id_2")      #graph information
edges=gn.edges

# Add edges to the graph
g.add_edges(edges)

out_fig_name = "graph.eps"

visual_style = {}

# Define colors used for outdegree visualization
colours = ['#fecc5c', '#a31a1c']

# Set bbox and margin
visual_style["bbox"] = (3000,3000)
visual_style["margin"] = 17

# Set vertex colours
visual_style["vertex_color"] = 'grey'

# Set vertex size
visual_style["vertex_size"] = 20

# Set vertex lable size
visual_style["vertex_label_size"] = 8

# Don't curve the edges
visual_style["edge_curved"] = False

# Set the layout
my_layout = g.layout_fruchterman_reingold()
visual_style["layout"] = my_layout

# Plot the graph
plot(g, out_fig_name, **visual_style)


# n_classes = 4

# bins = [[] for x in range(n_classes)]  

# musae_facebook_target = pd.read_csv('musae_facebook_target.csv', usecols = ['id','page_type'] )
# df = pd.DataFrame(musae_facebook_target)
# output0=df.loc[(df['page_type'] =='company')]
# bins[0]=list(output0['id'].values)
# output1=df.loc[(df['page_type'] =='government')]
# bins[1]=list(output1['id'].values)
# output2=df.loc[(df['page_type'] =='politician')]
# bins[2]=list(output2['id'].values)
# output3=df.loc[(df['page_type'] =='tvshow')]
# bins[3]=list(output3['id'].values)

# node_colours = []

# for i in range(n_vertices):
#     if i in bins[0]:
#         node_colours.append("yellow")
#     elif i in bins[1]:
#         node_colours.append("blue")
#     elif i in bins[2]:
#         node_colours.append("orange")
#     elif i in bins[3]:
#         node_colours.append("grey")
        

# out_fig_name = "labelled_graph.eps"

# g.vs["color"] = node_colours

# visual_style = {}

# # Define colors used for outdegree visualization
# colours = ['#fecc5c', '#a31a1c']

# # Set bbox and margin
# visual_style["bbox"] = (3000,3000)
# visual_style["margin"] = 17

# # Set vertex size
# visual_style["vertex_size"] = 20

# # Set vertex lable size
# visual_style["vertex_label_size"] = 8

# # Don't curve the edges
# visual_style["edge_curved"] = False

# # Set the layout
# visual_style["layout"] = my_layout

# # Plot the graph
# plot(g, out_fig_name, **visual_style)





