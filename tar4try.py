import networkx as nx
import matplotlib.pyplot as plt
import community
import collections
import numpy as np
import scipy as sp
from collections import Counter
from operator import itemgetter
import pandas as pd
import scipy.special
import seaborn as sns
from networkx.utils import powerlaw_sequence
from numpy import random

# יבוא דאטא ראשוני
dfile = pd.read_csv("musae_facebook_edges.csv")                        #load database from csv file
facebook = nx.from_pandas_edgelist(dfile, source="id_1", target="id_2")      #graph information
# Data = open('musae_facebook_edges.csv', "r")
# next(Data, None)  # skip the first line in the input file
# # Graphtype = nx.DiGraph()
# G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype, nodetype=int, data=(('weight', float),))
# original_nodes = list(G.nodes)
# original_edges = list(G.edges)
# print("number of nodes ", len(original_nodes))
# print("number of edges ", len(original_edges))
# print("original_nodes", original_nodes)
# print("original_edges", original_edges)

# ציור גרף מקורי
# color_map = []
# for node in G:
#     if node < 1000:
#         color_map.append('blue')  # news
#     elif node < 1000000:
#         color_map.append('red')  # articles
#     else:
#         color_map.append('green')  # users
# nx.draw(G, node_color=color_map, with_labels=False)
# pos = nx.spring_layout(G)
# plt.show()

# רכיבי קשירות
# print("num of strongly cc ", nx.number_strongly_connected_components(G))
# print("num of weakly cc ", nx.number_weakly_connected_components(G))
# # print("all the components")
# b = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
# print("b", len(b))

# ציור רכיב הקשירות הגדול ביותר
# largest = len(b[0])
# print("largest connected components ", largest)
# NH = G.subgraph(b[0])
# nodes = list(NH.nodes)
# edges = list(NH.edges)
# print("largest connected components nodes", len(nodes))
# print("largest connected components edges", len(edges))
# color_map = []
# for node in NH:
#     if node < 1000:
#         color_map.append('blue')  # news
#     elif node < 1000000:
#         color_map.append('red')  # articles
#     else:
#         color_map.append('green')  # users
# nx.draw(NH, node_color=color_map, with_labels=False)
# plt.show()


# התפלגות דרגות
degree_sequence = sorted([d for n, d in facebook.degree()], reverse=True)  # degree sequence
print("degree", degree_sequence)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
print("deg",deg)
p = []
for x in cnt:
    p.append(x / 22470)
print("cnt", cnt)
print("probabily", p)
fig, ax = plt.subplots()
plt.bar(deg, cnt, width=1, color="b")
plt.title("Degree Histogram")
plt.ylabel("count")
plt.xlabel("Degree")
# plt.xscale('log')
# plt.yscale('log')
plt.xlim([0, 800])
plt.ylim([0, 3000])
# plt.yticks(np.arange(0, 1, step=0.1))
plt.show()

sns.distplot(random.exponential(size=22470), hist=False)
# sns.distplot(random.binomial(degreeCount,p,len(nodes)), hist=False, color="r")
plt.show()
# supported values are 'linear', 'log', 'symlog', 'logit', 'function', 'functionlog'

#התפלגות עם נרמול של העמודות
# bins=np.logspace(np.log10(1), np.log10(1000), 50)
# print("vvvvvvvvvvvvvvvvvvvvvvvvvv",bins)
# plt.hist(deg, bins=np.logspace(np.log10(1), np.log10(1000), 50), density=True, stacked=True, edgecolor='black')
# plt.gca().set_xscale("log")
# plt.gca().set_yscale("log")
# plt.show()
#התפלגות מצטברת

cum=[]
cum.append(p[1])
print(cum)
for i in range(1,12):
    cum.append(cum[i-1]+p[i])
print("cumulative",cum)

plt.hist(deg, bins=np.logspace(np.log10(1), np.log10(1000), 50), density=True, stacked=True, edgecolor='black',
         cumulative=-1)
#plt.loglog(deg,cum,'ro-')
# plt.gca().set_xscale("log")
# plt.gca().set_yscale("log")
# plt.show()

#ניסויים שונים
# plt.hist(x, bins=np.logspace(start=np.log10(10), stop=np.log10(15), num=10))
# plt.gca().set_xscale("log")
# plt.show()
# x, bins, y=plt.hist(p, density=True)
# plt.title("Degree Histogram")
# plt.ylabel("probability")
# plt.xlabel("Degree")
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
#
# _, bins = np.histogram(np.log10(deg + 1), bins='auto')
# plt.hist(deg, bins=10**bins);
# plt.gca().set_xscale("log")
# plt.show()

# plt.plot(list(cumulative_sum(nx.degree_histogram(G))))


#גרף לינארי
deg2=deg[1:12]
cnt2=cnt[1:12]
p2=p[1:12]
x=np.array(deg)
y=np.array(cum)

#Applying a linear fit with .polyfit()
fit = np.polyfit(x,y,1)
ang_coeff = fit[0]
print("aaaaaaaaaaaaaaaaaaaaa ",ang_coeff)
intercept = fit[1]
print(fit)
fit_eq = ang_coeff*x + intercept  #obtaining the y axis values for the fitting function
#Plotting the data
poly1d_fn = np.poly1d(fit)


fig = plt.figure()
ax = fig.subplots()
plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k')
#ax.plot(x, fit_eq,color = 'r', alpha = 0.5, label = 'Linear fit')
ax.scatter(x,y,s = 5, color = 'b', label = 'Data points') #Original data points
ax.set_title('Linear fit ')
ax.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()

# #log log
# plt.figure()
# plt.loglog(range(len(degree_sequence)),degree_sequence,'ro-')
# plt.show()




#יצירת גרף אקראי בר
print("***********bar**************")
BAR= nx.barabasi_albert_graph(312,int(406/312))
bar_nodes = list(BAR.nodes)
bar_edges = list(BAR.edges)
print("number of nodes ", len(bar_nodes))
print("number of edges ", len(bar_edges))

nx.draw(BAR, with_labels=False)
plt.show()
# התפלגות דרגות
degree_sequence = sorted([d for n, d in BAR.degree()], reverse=True)  # degree sequence
print("degree", degree_sequence)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
print("deg",deg)
p = []
for x in cnt:
    p.append(x / 22470)
print("cnt", cnt)
print("probabily", p)
fig, ax = plt.subplots()
plt.bar(deg, p, width=1, color="b", edgecolor='black')
plt.title("Degree Histogram")
plt.ylabel("probability")
plt.xlabel("Degree")
plt.xscale('log')
plt.yscale('log')
plt.xlim([0, 400])
plt.ylim([0, 1])
# plt.yticks(np.arange(0, 1, step=0.1))
plt.show()

#התפלגות עם נרמול של העמודות
# plt.hist(deg, bins=np.logspace(np.log10(1), np.log10(1000), 50), density=True, stacked=True, edgecolor='black')
# plt.gca().set_xscale("log")
# plt.gca().set_yscale("log")
# plt.show()
#התפלגות מצטברת
# plt.hist(deg, bins=np.logspace(np.log10(1), np.log10(1000), 50), density=True, stacked=True, edgecolor='black',
#          cumulative=-1)
# plt.gca().set_xscale("log")
# plt.gca().set_yscale("log")
# plt.show()
cum=[]
cum.append(p[1])
print(cum)
for i in range(1,12):
    cum.append(cum[i-1]+p[i])
print("ccccccccccccccccccccccccccc",cum)

plt.hist(deg, bins=np.logspace(np.log10(1), np.log10(1000), 50), density=True, stacked=True, edgecolor='black',
         cumulative=-1)

x=np.array(deg)
y=np.array(cnt)

#Applying a linear fit with .polyfit()
fit = np.polyfit(x,y,1)
ang_coeff = fit[0]
intercept = fit[1]
print(fit)
fit_eq = ang_coeff*x + intercept  #obtaining the y axis values for the fitting function
print(fit_eq)
#Plotting the data
fig = plt.figure()
ax = fig.subplots()
ax.plot(x, fit_eq,color = 'r', alpha = 0.5, label = 'Linear fit')
ax.scatter(x,y,s = 5, color = 'b', label = 'Data points') #Original data points
ax.set_title('Linear fit ')
ax.legend()
# plt.xscale('log')
# plt.yscale('log')
plt.clr()
plt.show()