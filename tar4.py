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
import powerlaw
import community
import seaborn as sns
from numpy import random
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit


#load graph from csv file
dfile = pd.read_csv("musae_facebook_edges.csv")                        #load database from csv file

facebook = nx.from_pandas_edgelist(dfile, source="id_1", target="id_2")      #graph information
print(nx.info(facebook))
# Pdegree_sequence = sorted([d for n, d in facebook.degree()], reverse=True)
# fit = powerlaw.Fit(Pdegree_sequence, xmin=1) 
# fig2 = fit.plot_pdf(color='b', linewidth=2)
# fit.power_law.plot_pdf(color='g', linestyle='--', ax=fig2)

degree_sequenceF = sorted([d for n, d in facebook.degree()], reverse=True)  # degree sequence of source
degreeCount = collections.Counter(degree_sequenceF)
deg, cnt = zip(*degreeCount.items())

cnt_prop = []
for x in cnt:
    cnt_prop.append(x / 22470)
# print("cnt", cnt)
# print("probabily", cnt_prop)



#degree sequence in range of [0,1]
x=np.array(deg)
y=np.array(cnt)

maxDeg = degree_sequenceF[0]
print(maxDeg)
prop_sequence = [0] * len(degree_sequenceF) 
for i in degree_sequenceF:
    prop_sequence[i] = degree_sequenceF[i]/maxDeg
degreeCountProp = collections.Counter(prop_sequence)
degP, cntP = zip(*degreeCountProp.items())

xP=np.array(degP)
yP=np.array(cntP)

# linear X linear distribution
# plt.scatter(deg,cnt,color="red")
# linear_model=np.polyfit(x,cnt,1)
# linear_model_fn=np.poly1d(linear_model)
# #plot pit
# # plt.xscale('linear')
# # plt.xlim([0,maxDeg+10])
# # plt.ylim([0,4000])
# plt.yscale('linear')
# plt.xscale('linear')
# plt.xlabel('Degree')
# plt.ylabel('Count')
# plt.title('Facebook degree distribution')
# x_s=np.arange(0,720)
# plt.plot(x_s,linear_model_fn(x_s),color="green")
# plt.show()

# log X log degree distribution
# plt.scatter(np.log(x),np.log(cnt),color="red")
# linear_model=np.polyfit(np.log(x),np.log(cnt),1)
# linear_model_fn=np.poly1d(linear_model)
#plot pit
# plt.xscale('linear')
# plt.xlim([0,maxDeg+10])
# plt.ylim([0,4000])
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('Degree')
# plt.ylabel('Count')
# plt.title('Facebook degree distribution')
# x_s=np.arange(0,np.log(2000))
# plt.plot(x_s,linear_model_fn(x_s),color="green")
# plt.show()
# print(linear_model_fn)

# fig, ax = plt.subplots()
# plt.bar(deg, cnt, width=0.80, color="r")
# plt.xlabel('Degree')
# plt.ylabel('Number of nodes')
# plt.xscale('log')
# plt.show()


# linear X linear prop distribution
# plt.scatter(deg,cnt_prop,color="red")
# linear_model=np.polyfit(x,cnt_prop,1)
# linear_model_fn=np.poly1d(linear_model)
# #plot pit
# # plt.xscale('linear')
# # plt.xlim([0,maxDeg+10])
# # plt.ylim([0,4000])
# plt.yscale('linear')
# plt.xscale('linear')
# plt.xlabel('Degree')
# plt.ylabel('Probability')
# plt.title('Facebook degree distribution')
# x_s=np.arange(0,720)
# plt.plot(x_s,linear_model_fn(x_s),color="green")
# plt.show()


# log X log prop distribution
# plt.scatter(np.log(x),cnt_prop,color="red")
# linear_model=np.polyfit(np.log(x),cnt_prop,1)
# # plt.scatter(x,cnt_prop,color="red")
# # linear_model=np.polyfit(x,cnt_prop,1)
# linear_model_fn=np.poly1d(linear_model)
#plot pit
# plt.xscale('linear')
# plt.xlim([0,maxDeg+10])
# plt.ylim([0,4000])
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('Degree')
# plt.ylabel('Probability')
# plt.title('Facebook degree distribution')
# x_s=np.arange(0,np.log(2000))
# plt.plot(x_s,linear_model_fn(x_s),color="green")
# plt.show()
# print(linear_model_fn)

# log X linear prop distribution
# plt.scatter(deg,cnt_prop,color="red")
# linear_model=np.polyfit(x,cnt_prop,1)
# linear_model_fn=np.poly1d(linear_model)
# #plot pit
# # plt.xscale('linear')
# # plt.xlim([0,maxDeg+10])
# # plt.ylim([0,4000])
# plt.yscale('log')
# plt.xscale('linear')
# plt.xlabel('Degree')
# plt.ylabel('Probability')
# plt.title('Facebook degree distribution')
# x_s=np.arange(0,720)
# plt.plot(x_s,linear_model_fn(x_s),color="green")
# plt.show()

# sns.distplot(random.exponential(size=22470), hist=False)
# plt.show()

# log on bins log X log
# bins=np.logspace(np.log10(1), np.log10(1000), 200)
# bins=np.logspace(1, 1000, 50)

# print("vvvvvvvvvvvvvvvvvvvvvvvvvv",bins)
# plt.scatter(deg,cnt_prop,color="red")
# plt.scatter(np.log(x),np.log(cnt),color="red")
linear_model=np.polyfit(np.log(x),np.log(cnt),1)
linear_model_fn=np.poly1d(linear_model)
print('i fn', linear_model_fn)
print('i x', x)
# m, b = np.polyfit(x, cnt_prop, 1)
# print(m)

# ya = np.polyval(linear_model, x)
# plt.hist(np.log(x), bins=np.logspace(np.log10(1), np.log10(720), 50), density=True, stacked=True, edgecolor='red')
plt.hist(x, bins=np.logspace(np.log10(1), np.log10(1000), 50), density=True, stacked=True, edgecolor='red', alpha=0.3)
# plt.hist(x, bins=np.logspace(1, 720, 50), density=True, stacked=True, edgecolor='red')
# plt.hist(np.log10(x), bins=20, density=True, stacked=True, edgecolor='red')
# plt.gca().set_xscale("log")
# # plt.gca().set_xscale("log")
# plt.gca().set_yscale("linear")
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Facebook degree distribution')
x_s=np.arange(0, np.log(720))
# plt.ylim([0,0.01])
plt.plot(x_s,linear_model_fn(x_s),color="green")
# plt.plot(x, y)
# plt.plot(x, ya)
# plt.plot(x, m*x + b)
plt.show()



#cummulative distribution
# cum=[]
# print(len(deg))
# cum.append(cnt_prop[1])
# print(cum)
# for i in range(1,len(deg)):
#     cum.append(cum[i-1]+cnt_prop[i])
# # print("cum",cum)

# plt.hist(deg, bins=np.logspace(np.log10(1), np.log10(1000), 50), density=True, stacked=True, edgecolor='black',
#          cumulative=-1)
# m, b = np.polyfit(x, cnt_prop, 1)
# print(m)
# plt.ylim([0,1])
# plt.xlabel('Degree')
# plt.ylabel('Count')
# plt.title('Facebook Cumulative distribution')
# plt.plot(x, m*x + b)

# plt.show()
# print(m)


# define the true objective function
# def objective(x, a, b):
# 	return a * x + b

# popt, _ = curve_fit(objective, x, y)
# # summarize the parameter values
# a, b = popt
# print('y = %.5f * x + %.5f' % (a, b))
# # plot input vs output
# plt.scatter(x, y)
# # define a sequence of inputs between the smallest and largest known inputs
# x_line = arange(min(x), max(x), 1)
# # calculate the output for the range
# y_line = objective(x_line, a, b)
# # create a line plot for the mapping function
# plt.plot(x_line, y_line, '--', color='red')
# plt.show()


# barabasiF = nx.barabasi_albert_graph(22470, 171002//22470)
# print("barabasi info:")
# print(nx.info(barabasiF))


# degree_sequenceBara = sorted([d for n, d in barabasiF.degree()], reverse=True)  # degree sequence of source
# degreeCountBara = collections.Counter(degree_sequenceBara)
# degBara, cntBara = zip(*degreeCountBara.items())

# cnt_prop_bara = []
# for k in cntBara:
#     cnt_prop_bara.append(k / 22470)
# print("cnt", cnt)
# print("probabily", cnt_prop)



# #degree sequence in range of [0,1]
# xBara=np.array(degBara)
# yBara=np.array(cntBara)

# maxDegBara = degree_sequenceBara[0]
# print(maxDegBara)

# linear X linear distribution
# plt.scatter(degBara,cntBara,color="red")
# linear_modelBara=np.polyfit(xBara,cntBara,1)
# linear_model_fn_bara=np.poly1d(linear_modelBara)

# # linear x linear
# plt.yscale('linear')
# plt.xscale('linear')
# plt.xlabel('Degree')
# plt.ylabel('Count')
# plt.title('Barabasi degree distribution')
# x_s=np.arange(0,1000)
# plt.plot(x_s,linear_model_fn_bara(x_s),color="green")
# plt.show()
# print(linear_model_fn_bara)

# log x log
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('Degree')
# plt.ylabel('Count')
# plt.title('Barabasi degree distribution')
# x_s=np.arange(0,1000)
# plt.plot(x_s,linear_model_fn_bara(x_s),color="green")
# plt.show()

# log X linear
# plt.yscale('log')
# plt.xscale('linear')
# plt.xlabel('Degree')
# plt.ylabel('Count')
# plt.title('Barabasi degree distribution')
# x_s=np.arange(0,1000)
# plt.plot(x_s,linear_model_fn_bara(x_s),color="green")
# plt.show()


# plt.scatter(degBara,cnt_prop_bara,color="red")
# plt.scatter(xBara,cnt_prop_bara,color="red")
# plt.scatter(np.log(xBara),np.log(cnt_prop_bara),color="red")
# # linear_model_bara=np.polyfit(xBara,cnt_prop_bara,1)
# linear_model_bara=np.polyfit(np.log10(xBara),np.log10(cnt_prop_bara),1)
# linear_model_fn_bara=np.poly1d(linear_model_bara)
# log X linear prop distribution
# plt.yscale('linear')
# plt.xscale('linear')
# plt.xlabel('Degree')
# plt.ylabel('Probability')
# plt.title('Barabasi degree distribution')
# x_s=np.arange(0,720)
# plt.plot(x_s,linear_model_fn_bara(x_s),color="green")
# plt.show()

# print(linear_model_fn_bara)

# log X log prop distribution
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('Degree')
# plt.ylabel('Probability')
# plt.title('Barabasi degree distribution')
# x_s=np.arange(0,np.log(2000))
# plt.plot(x_s,linear_model_fn_bara(x_s),color="green")
# plt.show()

# print(linear_model_fn_bara)

# log on bins log X log
# binsBara=np.logspace(np.log10(1), np.log10(1000), 50)


# # plt.scatter(np.log10(deg),np.log10(cnt_prop),color="red")
# linear_model_bara=np.polyfit(xBara,np.log(cnt_prop_bara),1)
# # linear_model_bara=np.polyfit(xBara,cnt_prop_bara,1)
# # mBara, bBara = np.polyfit(xBara, cnt_prop_bara, 1)
# # mBara, bBara = np.polyfit(np.log10(xBara), np.log10(cnt_prop_bara), 1)
# # mBara, bBara = np.polyfit(xBara,cnt_prop_bara, 1)
# print(mBara)
# linear_model_fn=np.poly1d(linear_model_bara)
# ya = np.polyval(linear_model_bara, xBara)
# plt.hist(degBara, bins=np.logspace(np.log10(1), np.log10(1000), 50), density=True, stacked=True, edgecolor='red', alpha=0.3)
# plt.gca().set_xscale("log")
# plt.gca().set_yscale("log")
# plt.xlabel('Degree')
# plt.ylabel('Count')
# plt.title('Barabasi degree distribution')
# x_s=np.arange(0,709)
# plt.plot(x_s,linear_model_fn(x_s),color="green")
# plt.plot(xBara, mBara*xBara + bBara, color="red")
# plt.plot(x, ya)
# plt.plot(xBara, mBara*xBara + bBara)
# plt.scatter(deg,cnt_prop_bara,color="red")
# plt.show()
# print(linear_model_fn(x_s))
# cummulative distribution
# cumBara=[]
# print(len(degBara))
# cumBara.append(cnt_prop_bara[1])
# print(cumBara)
# for i in range(1,len(degBara)):
#     cumBara.append(cumBara[i-1]+cnt_prop_bara[i])
# # print("cum",cum)

# plt.hist(degBara, bins=np.logspace(np.log10(1), np.log10(1000), 50), density=True, stacked=True, edgecolor='black',
#          cumulative=-1)
# mBara, bBara = np.polyfit(xBara, cnt_prop_bara, 1)
# print(mBara)
# plt.ylim([0,1])
# plt.xlabel('Degree')
# plt.ylabel('Count')
# plt.title('Barabasi Cumulative distribution')
# plt.plot(xBara, mBara*xBara + bBara)
# plt.show()

# print(mBara)