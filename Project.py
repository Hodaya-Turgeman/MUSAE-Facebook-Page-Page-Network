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
import json
import ast
from decimal import *


'''##################################################
Context
This webgraph is a page-page graph of verified Facebook sites.
Nodes represent official Facebook pages while the links are mutual likes between sites. 
Node features are extracted from the site descriptions that the page owners created to summarize the purpose of the site. This graph was collected through the Facebook Graph API in November 2017 and restricted to pages from 4 categories which are defined by Facebook. 
These categories are: politicians, governmental organizations, television shows, and companies. The task related to this dataset is multi-class node classification for the 4 site categories.

Content - about the graphs
Directed: No.
Node features: Yes.
Edge features: No.
Node labels: Yes. Binary-labeled.
Temporal: No.
###############################################################'''



#load database from musae_facebook_edges.csv file
df = pd.read_csv("musae_facebook_edges.csv")                            
# Build a graph if all the nodes 
g = nx.from_pandas_edgelist(df, source="id_1", target="id_2")      
#graph g information
print('*******************************************************')
print('graph g information',nx.info(g))
print( 'The number Connected Components in graph g : {0}'.format(nx.number_connected_components(g))) 

#load database from musae_facebook_target.csv file
musae_facebook_target = pd.read_csv('musae_facebook_target.csv', usecols = ['id','page_type'] )
#Parser the informetion in musae_facebook_target.csv to DataFrame
df = pd.DataFrame(musae_facebook_target)

government=df.loc[(df['page_type'] =='government') ]#Data of all goverment in the file musae_facebook_target.csv
politician=df.loc[(df['page_type'] =='politician') ]#Data of all politician in the file musae_facebook_target.csv
tvshow=df.loc[(df['page_type'] =='tvshow') ]#Data of all tvshow in the file musae_facebook_target.csv
company=df.loc[(df['page_type'] =='company') ]#Data of all company in the file musae_facebook_target.csv

gPolitician=g.subgraph(list(politician['id'].values))#gPolitician is a graph of all politician in database 
gTvshow=g.subgraph(list(tvshow['id'].values))#gTvshow is a graph of all tvshow in database
gGovernment=g.subgraph(list(government['id'].values))#gGovernment is a graph of all government in database
gCompany=g.subgraph(list(company['id'].values))#gCompany is a graph of all company in database

#Calculate for each node in graph g its betweenness and  saved in file fileBetweenness.txt
# betweenness=nx.betweenness_centrality(g)
# with open('fileBetweenness.txt', 'w') as file:
#      file.write(json.dumps(betweenness))

#Calculate for each node in graph g its closeness_centrality and  saved in file closeness_centrality.txt
# closeness_centrality = nx.closeness_centrality(g)
# with open('closeness_centrality.txt', 'w') as file:
#      file.write(json.dumps(close_centrality))

#Calculate for each node in graph g its degree_centrality and  saved in file degree_centrality.txt
# degree_centrality=nx.degree_centrality(g)
# with open('degree_centrality.txt', 'w') as file:
#      file.write(json.dumps(degree_centrality))


'''################################################################################

Exploring the strong vertices The community of politicians

#####################################################################################'''

#graph gPolitician information
print('*******************************************************')
print('graph gPolitician information',nx.info(gPolitician))
print( 'The number Connected Components in graph gPolitician : {0}'.format(nx.number_connected_components(gPolitician))) 

#Distribution of the degree of politicians' graphs
degree_sequencePolitician = sorted([d for n, d in gPolitician.degree()], reverse=True) 
in_histS = range(len(degree_sequencePolitician))
plt.figure() 
plt.grid(True)
plt.loglog(degree_sequencePolitician,in_histS,color='navy',marker='o') # source graph degree
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Politicians degree distribution')
plt.show()

#Find the largest binding component in the politicians' graph
s = sorted(nx.connected_components(gPolitician), key=len, reverse=True)
gPoliticianGiant = gPolitician.subgraph(s[0])
print('*******************************************************')
print('gPoliticianGiant',nx.info(gPoliticianGiant))
print( 'The number Connected Components gPoliticianGiant: {0}'.format(nx.number_connected_components(gPoliticianGiant))) 


#Finding the strongest apex within the community of politicians
max_grade = max(gPolitician.degree, key=lambda x: x[1])[1]
strongInPolitician=  [node[0] for node in gPolitician.degree if node[1] == max_grade]    #Node with the largest degree
strongInPolitician= strongInPolitician[0]
print("The strongest apex within the community of politicians",strongInPolitician,"Internal  degree:",max_grade,'External degree:',len(list(g.neighbors(strongInPolitician)))-max_grade,'Overall degree:',len(list(g.neighbors(strongInPolitician))))

#Finding the strongest apex outside the community of politicians That is, the politician most linked to outside pages

#Building a graph of the communitys Tvshow ,Government and  Company( not Politician )
g_Tvshow_Government_Company=g.subgraph(list(government['id'].values)+list(tvshow['id'].values)+list(company['id'].values))
#Graph where all the communities are and there are arcs of politicians if the rest of the communities but there are no arcs between the communities themselves
gPolitician_and_Other= nx.Graph()
gPolitician_and_Other.add_nodes_from(g.nodes)
gPolitician_and_Other.add_edges_from(g.edges-g_Tvshow_Government_Company.edges-gPolitician.edges)

pageType = {}
for node in gPolitician_and_Other.nodes:
    if node in gPolitician.nodes :
        pageType[node]='politician'
    elif node in gTvshow.nodes:
        pageType[node]='tvshow'
    elif node in gGovernment.nodes:
        pageType[node]='government'
    else:
        pageType[node]='company'
nx.set_node_attributes(gPolitician_and_Other, pageType, "page_type")
#Finding the strongest apex outside the community of politicians 
degree_sequenceF = sorted([d for n, d in gPolitician_and_Other.degree()], reverse=True)  # degree sequence of source
for i in degree_sequenceF:
    strongOutPolitician= [node[0] for node in gPolitician_and_Other.degree if node[1] == i]    #Node with the largest degree
    strongOutPolitician = strongOutPolitician[0]
    if gPolitician_and_Other.nodes[strongOutPolitician]['page_type']=='politician':
        break
print("The strongest apex outside the community of politicians That is, the politician most linked to outside pages",strongOutPolitician,"Internal  degree:",len(list(gPolitician.neighbors(strongOutPolitician))),'External degree:',i,'Overall degree:',len(list(g.neighbors(strongOutPolitician))))

#Finding the politician with the highest betweeness out The community of politicians

# bitPoliticianGiant = nx.betweenness_centrality(gPoliticianGiant)
# resBitHighestPolitician = dict(sorted(bitPoliticianGiant.items(), key = itemgetter(1), reverse = True)[:1])
# print("The  politician with the highest betweeness out The community of politicians  is " + str(resBitHighestPolitician),"Internal  degree:",len(list(gPolitician.neighbors(resBitHighestPolitician))),'Overall degree:',len(list(g.neighbors(resBitHighestPolitician))))

#Draws the largest binding element in the politicians' graph so that the strongest nodes are highlighted

color_map = []
node_sizes=[]
alpha_sizes=[]
for node in gPoliticianGiant.nodes:
    if(node== strongInPolitician ):#The strongest apex within the community of politicians in navy color in the gragh
        color_map.append('navy') 
        node_sizes.append(1000)
        alpha_sizes.append(1)
    elif node == strongOutPolitician :#The strongest apex outside the community of politicians in teal color in graph
        color_map.append('teal') 
        node_sizes.append(1000)
        alpha_sizes.append(1)
    else:
        alpha_sizes.append(0.5)#other node in gray color
        color_map.append('gray') 
        node_sizes.append(100)
# nx.draw(gPoliticianGiant,node_color=color_map,node_size=node_sizes,alpha=alpha_sizes, with_labels=False)
# plt.show()

#####################Central indices of strongest nodes in the community of politicians####################

#Closeness Centrality of strongest nodes in the community of politicians 
sumClose=[0.25144641278438656,0.3164380474889446]
fig, ax = plt.subplots()
deg=['Manfred Weber','Barack Obama']
colors=["navy","teal"]
plt.barh(deg,sumClose, color=colors)
plt.title("Closeness Centrality in the community of politicians")
plt.ylabel("Nodes")
plt.xlabel("Closeness")
plt.show()

#Betweness Centrality of strongest nodes in the community of politicians 
sumBetweness=[0.005395128910842675,0.08962832614191246]
fig, ax = plt.subplots()
deg=['Manfred Weber','Barack Obama']
colors=["navy","teal"]
plt.barh(deg,sumBetweness, color=colors)
plt.title("Betweenness Centrality in the community of politicians")
plt.ylabel("Nodes")
plt.xlabel("Betweenness")
plt.show()

#Degree Centrality of strongest nodes in the community of politicians 
sumDegree=[0.014508878899817527,0.015176465352263118]
fig, ax = plt.subplots()
deg=['Manfred Weber','Barack Obama']
colors=["navy","teal"]
plt.barh(deg,sumDegree, color=colors)
plt.title("Degree Centrality in the community of politicians")
plt.ylabel("Nodes")
plt.xlabel("Degree Centrality")
plt.show()


'''###############################################################

Exploring the strong vertices The community of government

###################################################################'''

#graph gGovernment information
print('*******************************************************')
print('graph gGovernment information',nx.info(gGovernment))
print( 'The number Connected Components in graph gGovernment : {0}'.format(nx.number_connected_components(gGovernment))) 


#Distribution of the degree of Government's graphs
degree_sequenceGovernment = sorted([d for n, d in gGovernment.degree()], reverse=True) 
in_histS = range(len(degree_sequenceGovernment))
plt.figure() 
plt.grid(True)
plt.loglog(degree_sequenceGovernment,in_histS,color='gold',marker='o') # source graph degree
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Government degree distribution')
plt.show()

#Find the largest binding component in the Government's graph
s = sorted(nx.connected_components(gGovernment), key=len, reverse=True)
gGovernmentGiant = gGovernment.subgraph(s[0])
print('*******************************************************')
print('gGovernmentGiant',nx.info(gGovernmentGiant))
print( 'The number Connected Components gGovernmentGiant: {0}'.format(nx.number_connected_components(gGovernmentGiant))) 


#Find the largest binding component in the Governments' graphs
degree_sequenceGovernment = sorted([d for n, d in gGovernment.degree()], reverse=True)
strongInGovernment = [node[0] for node in gGovernment.degree if node[1] ==degree_sequenceGovernment[0]]
strongInGovernment= strongInGovernment[0]
print("The strongest apex within the community of government",strongInGovernment,"Internal  degree:",degree_sequenceGovernment[0],'External degree:',len(list(g.neighbors(strongInGovernment)))-degree_sequenceGovernment[0],'Overall degree:',len(list(g.neighbors(strongInGovernment))))

#Finding the strongest apex outside the community of government That is, the government most linked to outside pages

#Building a graph of the communitys Tvshow ,Politicians and  Company( not government )
g_Tvshow_Politician_Company=g.subgraph(list(politician['id'].values)+list(tvshow['id'].values)+list(company['id'].values))
#Graph where all the communities are and there are arcs of government if the rest of the communities but there are no arcs between the communities themselves
gGovernment_and_Other= nx.Graph()
gGovernment_and_Other.add_nodes_from(g.nodes)
gGovernment_and_Other.add_edges_from(g.edges-g_Tvshow_Government_Company.edges-gGovernment.edges)
pageType = {}
for node in gGovernment_and_Other.nodes:
    if node in gPolitician.nodes :
        pageType[node]='politician'
    elif node in gTvshow.nodes:
        pageType[node]='tvshow'
    elif node in gGovernment.nodes:
        pageType[node]='government'
    else:
        pageType[node]='company'
nx.set_node_attributes(gGovernment_and_Other, pageType, "page_type")

#Finding the strongest apex outside the community of government 
degree_sequenceG = sorted([d for n, d in gGovernment_and_Other.degree()], reverse=True)  # degree sequence of source
for i in degree_sequenceG:
    strongOutGovernment= [node[0] for node in gGovernment_and_Other.degree if node[1] == i]    #Node with the largest degree
    strongOutGovernment = strongOutGovernment[0]
    if gGovernment_and_Other.nodes[strongOutGovernment]['page_type']=='government' :
        break
print("The strongest apex outside the community of government That is, the government most linked to outside pages",strongOutGovernment,"Internal  degree:",len(list(gGovernment.neighbors(strongOutGovernment))),'External degree:',i,'Overall degree:',len(list(g.neighbors(strongOutGovernment))))
    
#Finding the government with the highest betweeness out The community of government
resBitHighestGovernment=10379
# bitGovernmentGiant = nx.betweenness_centrality(gGovernmentGiant)
# resBitHighestGovernment = dict(sorted(bitGovernmentGiant.items(), key = itemgetter(1), reverse = True)[:1])
# print("The  government with the highest betweeness out The community of government  is " + str(resBitHighestGovernment),"Internal  degree:",len(list(gGovernment.neighbors(resBitHighestGovernment))),'Overall degree:',len(list(g.neighbors(resBitHighestGovernment))))

#Draws the largest binding element in the government's graph so that the strongest nodes are highlighted

color_map = []
node_sizes=[]
alpha_sizes=[]
for node in gGovernmentGiant.nodes:
    if(node== strongInGovernment ):#The strongest apex within the community of government in gold color in the gragh
        color_map.append('gold') 
        node_sizes.append(1000)
        alpha_sizes.append(1)
    elif node == resBitHighestGovernment :
        color_map.append('darkorange') # The government with the highest betweeness out The community of government in darkorange color in graph
        node_sizes.append(1000)
        alpha_sizes.append(1)
    elif node== strongOutGovernment:
        color_map.append('khaki') #The strongest apex outside the community of government in khaki color in graph
        node_sizes.append(1000)
        alpha_sizes.append(1)
    else:
        alpha_sizes.append(0.3)
        color_map.append('gray') #other node in gray color
        node_sizes.append(100)
# nx.draw(gGovernmentGiant,node_color=color_map,node_size=node_sizes,alpha=alpha_sizes, with_labels=False)
# plt.show()



# sumClose=[0.2989011866120364,0.23110548835678432,0.30202704519181656]

# fig, ax = plt.subplots()
# deg=['US Army','Senate of Canada','US Department of State']
# colors=["gold","khaki","darkorange"]
# # labels=['U.S. Army','U.S. Army Chaplain Corps','The White House','The Obama White House','Senate of Canada - Sénat du Canada','European Parliament']
# # # for i in range(0,6):
# # #     plt.barh(deg[i], sumClose[i], color=colors[i],label=labels[i])
# plt.barh(deg,sumClose, color=colors)
# plt.title("Closeness Centrality")
# plt.ylabel("Nodes")
# plt.xlabel("Closeness")
# # plt.legend()
# plt.show()

# print('gGovernment',nx.info(gGovernment))
# print( 'The number Connected Components gGovernment: {0}'.format(nx.number_connected_components(gGovernment))) 


# sumBetweness=[0.015054976806517162,0.0027104479220805507,0.015456013941175314]
# fig, ax = plt.subplots()
# deg=['US Army','Senate of Canada','US Department of State']
# colors=["gold","khaki","darkorange"]

# plt.barh(deg,sumBetweness, color=colors)
# plt.title("Betweenness Centrality")
# plt.ylabel("Nodes")
# plt.xlabel("Betweenness")
# plt.show()

sumDegreeCentrality=[0.0315545863189283,0.01183853309003516,0.02082869731630246]
# fig, ax = plt.subplots()
deg=['US Army','Senate of Canada','US Department of State']
colors=["gold","khaki","darkorange"]

plt.barh(deg,sumDegreeCentrality, color=colors)
plt.title("Degree Centrality")
plt.ylabel("Nodes")
plt.xlabel("Degree Centrality")
plt.show()

# sumDegree=[709,650,678,659,266,417]
# fig, ax = plt.subplots()
# deg=['G','H','I','J','K','L']
# colors=["gold","goldenrod","tan","navajowhite","darkorange","khaki"]
# plt.barh(deg,sumDegree, color=colors)
# plt.title("Degree")
# plt.ylabel("Nodes")
# plt.xlabel("Degree")
# plt.show()

# strong=[strongFirstInGovernment,strongSecondInGovernment,strongThirdInGovernment,strongFourInGovernment,16052,21120]
# gStrongGovernment=gGovernment.subgraph(list(gGovernment.neighbors(strongFirstInGovernment))+list(gGovernment.neighbors(strongSecondInGovernment))+list(gGovernment.neighbors(strongThirdInGovernment))+list(gGovernment.neighbors(strongFourInGovernment))+list(gGovernment.neighbors(16052))+list(gGovernment.neighbors(21120))+strong)

# color_map = []
# alpha_sizes=[]
# for node in gStrongGovernment.nodes:
#     if(node== strongFirstInGovernment ):
#         color_map.append('gold') 
#         alpha_sizes.append(1)
#     elif node == strongSecondInGovernment :
#         color_map.append('goldenrod') 
#         alpha_sizes.append(1)
#     elif node == strongThirdInGovernment:
#         color_map.append('tan') 
#         alpha_sizes.append(1)
#     elif node== strongFourInGovernment:
#         color_map.append('navajowhite')
#         alpha_sizes.append(1)
#     elif node==16052:
#         color_map.append('darkorange')
#         alpha_sizes.append(1)
#     elif node==21120:
#         color_map.append('khaki')
#         alpha_sizes.append(1)
#     else:
#         color_map.append('gainsboro')
#         alpha_sizes.append(0.3)

# degStrongGovernment=nx.degree_centrality(gStrongGovernment)
# node_sizes=[]
# for x in degStrongGovernment.values():
#      node_sizes.append(x*10000)
# print(nx.info(gStrongGovernment))
# print( 'The number Connected Components: {0}'.format(nx.number_connected_components(gStrongGovernment))) 
# nx.draw(gStrongGovernment,node_color=color_map,node_size=node_sizes,alpha=alpha_sizes,with_labels=False)
# plt.show()


# file = open("betweeness.txt", "r")

# contents = file.read()
# dictionary = ast.literal_eval(contents)

# file.close()
# print(dictionary)
# betwweeness=dict(sorted(dictionary.items(), key=lambda item: item[1]))
# max=-1
# min=1
# indexmax=-1
# # print(dictionary)
# a=gPolitician.nodes
# # print(type(dictionary))
# i=0

# while max<min:
#     if dictionary[i] >max:
#         max=dictionary[i]
#     if i in a:
#         min =max
#         indexmax=i
#         break
#     i=i+1

# # for i in dictionary:
# #     print(i)
# #     if i[0] in a:
# #         if Decimal(i[1]) > max:
# #             max=i[1]
# #             indexmax=i[0]

# print(indexmax,max)


# d=gPoliticianGiant.remove_nodes_from(a)
# print(d,nx.info(d))

# a=list(politician['id'].values)
# a.remove(11003)
# a.remove(14650)
# print(nx.info(gPoliticianGiant))
# gPolitician1=gPoliticianGiant.subgraph(a)
# s = sorted(nx.connected_components(gPolitician1), key=len, reverse=True)
# gPoliticianGiant1 = gPolitician1.subgraph(s[0])
# print(nx.info(gPolitician1))
# print( 'The number Connected Components gPoliticianGiant1: {0}'.format(nx.number_connected_components(gPolitician1))) 
# color_map=[]
# for i in range(len(gPolitician1.nodes) ):
#     color_map.append('teal')
# nx.draw(gPolitician1, node_color='navy',with_labels=False)
# plt.show()


# a=list(g.nodes)
# a.remove(11003)
# a.remove(14650)
# g1=g.subgraph(a)
# # a.remove(14650)
# t = []
# t1 = []
# t2 = [] 
# for node in g1.nodes:
#     if(node in gPolitician.nodes ):
#         pageType[node]=1
#         t1.append(node)
#     else:
#         t2.append(node) 
#         pageType[node]=2
# nx.set_node_attributes(g1, pageType, "page_type")
# t = [t1, t2] 
# print(nx.algorithms.community.modularity(g1 ,t,weight='None'))






# a=list(government['id'].values)
# a.remove(16052)
# a.remove(10379)
# a.remove(16895)
# print('gGovernmentGiant',nx.info(gGovernmentGiant))
# gGoverment1=gGovernmentGiant.subgraph(a)
# print('gGovernment1',nx.info(gGoverment1))
# print( 'The number Connected Components gGovernment1: {0}'.format(nx.number_connected_components(gGoverment1))) 
# s = sorted(nx.connected_components(gGoverment1), key=len, reverse=True)
# gGovernmentGiant1 = gGoverment1.subgraph(s[0])
# print("The max gain ",nx.info(gGovernmentGiant1))

# color_map=[]
# # for i in range(len(gGoverment1.nodes) ):
# #     color_map.append('gold')
# nx.draw(gGoverment1, node_color='gray',with_labels=False)
# plt.show()

# a=list(g.nodes)
# a.remove(10379)
# a.remove(16895)
# a.remove(16052)
# g1=g.subgraph(a)

# t = []
# t1 = []
# t2 = [] 
# for node in g1.nodes:
#     if(node in gGovernment.nodes ):
#         pageType[node]=1
#         t1.append(node)
#     else:
#         t2.append(node) 
#         pageType[node]=2
# nx.set_node_attributes(g1, pageType, "page_type")
# t = [t1, t2] 
# print(nx.algorithms.community.modularity(g1 ,t,weight='None'))
















































