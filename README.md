# MUSAE-Facebook-Page-Page-Network
# Project in Social Networks 

## Introduction
Today, social networks have gained tremendous momentum, becoming a central tool for hundreds of millions of people around the world to share and consume information.
For our project we took a graph of the world's largest network facebook.
Which contained official pages from 4 categories, we focused on the pages of politicians and pages of government institutions, our goal was to examine the structure of these communities, who are the strongest internal vertices in each community and who are the strongest external vertices in each community.
And finally understand how much these vertices have an impact on the community.

## DataSet
The data set we studied contains official Facebook pages from 4 categories:
Politicians, Government, Companies and TV shows.
The data contains 22,470 vertices which represent official facebook pages and
171,002 ribs which represent a mutual like between 2 pages meaning the graph is undirected.
In addition, each vertex has:
ID, facebook ID, page name, page type (Politicians/Government/Companies/TV shows)
The graph also contains one binding element and can be seen in the link:
https://www.kaggle.com/rozemberczki/musae-facebook-pagepage-network?select=musae_facebook_edges.csv

## Research Question
In our study we focused on 2 key communities that are indirectly related to each other which are politicians and government and we wanted to examine whether the strongest vertices( based on the degree and betweenness measures) from the outside have the most influence within the community or whether the strongest vertices from within have the impact.

## Workflow
#### We first built from the central graph the graph of the community of politicians and the graph of the government community.

#### We have cleared the unnecessary information from each community graph and worked on the largest binding component in each of the communities (all actions we will perform below will be either the largest binding component in each community or the largest graph of all communities combined)

#### Calculating the rank graph of the community graph.

#### Finding the strongest vertex within the community based on the rank index-That is, find the highest-ranking vertex within the graph.

#### Finding the strongest vertex outside the community based on the rank index-meaning, finding the vertex from the community that has the most arcs to vertices from other communities.

#### Finding the vertex in the community that has the highest betweenness.

#### Examining the bonding of the graph without any of the strong vertices to see how much of an impact they have within the graph, and  comparing between them. Our intuition was that if without the vertex our graph breaks down into more binding elements then this vertex has high importance and impact on the topology and bonding of the community.

#### Examining the modularity of the large graph with the strong vertices and without any of the strong vertices to see if they affected the modularity of the large graph also and  the topology of their communities, and comparing between them.

#### Checking additional centrality indices of each of the strongest vertices we found and comparing between them: Closeness, Betweenness and Degree.




## Conclusions
We came to a number of conclusions about the networks we researched.
We saw that the main differences between the networks were the number of arcs in the politician graph, there were much fewer arcs and relatively the same number of vertices compared to the government graph which contained many more arcs and then there are many more possible paths between vertices.

The distribution of power in the network according to the distribution of ranks graph and the size of the maximum rank. We saw that in the politicians' graph the power is concentrated in a smaller amount of vertices compared to the government graph in which the power is spread over more vertices.

This made us realize that the influence of the strongest vertices also depends very much on the structure of the community because in the politicians' graph we saw that the vertex that has the highest external rank and the highest betweenness greatly influenced the network's topology and its ties than the vertex with the highest internal ranking.

In the government graph, on the other hand, we saw that the vertex with the highest external rank and the vertex with the highest Betweenness did not affect the internal community so much, and the strongest inner vertex had a slightly greater effect, but not significantly, which led us to the conclusion that the effect of the strong vertices depends first on the structure and topology of the network.

When the community itself is connected with many paths meaning there are a lot of arks in the community and there are a number of powerful vertices within the network the strongest nodes have less influence because even without them you can reach each other and they do not form bridges but when the community do not have many arcs between the vertices within the community and the power is held in a smaller number of vertices So there is much more influence and power to the strong vertices in the network, especially the vertices that are externally strong and those with the highest betweenness in the community.
