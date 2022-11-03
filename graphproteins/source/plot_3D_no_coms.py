import networkx as nx
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cdlib import algorithms, evaluation
from plotly.offline import init_notebook_mode, iplot
from networkx.algorithms.community.centrality import girvan_newman
import community as community_louvain
import pandas as pd
import plotly.graph_objects as go

CUTOFF = 0.55

file = 'distance_matrix_full_skip3.csv'
file = 'distance_matrix_trajectories_only_skip3.csv'
file = 'distance_matrix.csv'
file = 'distance_matrix_chi3.csv'
file = 'distance_matrix_with_originals.csv'
file = '.distance_matrix_0930.csv'

dist_matrix = np.genfromtxt(file, delimiter=',', skip_header=1)
header = np.genfromtxt(file, delimiter=',', dtype=str, max_rows=1)
mean = np.mean(dist_matrix)
std = np.std(dist_matrix)
dist_matrix = (dist_matrix ) / std
#dist_mask = np.where(dist_matrix > 0.8, 0, dist_matrix)
dist_mask = np.where(dist_matrix > CUTOFF, 0, dist_matrix)
#dist_mask = 1 - dist_mask / np.max(dist_matrix)

#dist_mask = np.where(dist_matrix > 1.5, np.max(dist_matrix), dist_matrix)
#dist_mask = 1 - dist_mask / np.max(dist_matrix)

G = nx.from_numpy_array(dist_mask)
mapping =  { x:ind for  x, ind in enumerate(header) }
G_relab = nx.relabel_nodes(G, mapping)
pos = nx.spring_layout(G_relab)

louvain_coms = algorithms.louvain(G, weight='weight', resolution=1.2)
partition = community_louvain.best_partition(G_relab)
print("Mod Score: " + str(community_louvain.modularity(partition, G_relab)))
print("Mod Score: " + str(louvain_coms.newman_girvan_modularity()))


spring_3D = nx.spring_layout(G_relab,dim=3, seed = 1)
#print(spring_3D)
x_nodes = [spring_3D[i][0] for i in G_relab]# x-coordinates of nodes
y_nodes = [spring_3D[i][1] for i in G_relab]# y-coordinates
z_nodes = [spring_3D[i][2] for i in G_relab]# z-coordinates
edge_list = G_relab.edges()


x_edges=[]
y_edges=[]
z_edges=[]

#need to fill these with all of the coordiates
for edge in edge_list:
    #format: [beginning,ending,None]
    x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
    x_edges += x_coords

    y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
    y_edges += y_coords

    z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
    z_edges += z_coords
    
#partition 
community_label = [partition[i] for i in partition]
louvain_coms.to_node_community_map()
print(louvain_coms)
community_label = [0 for i in louvain_coms.to_node_community_map()]
for i in louvain_coms.to_node_community_map():
    community_label[i] = louvain_coms.to_node_community_map()[i][0]
names = [i for i in partition]



#create a trace for the edges
trace_edges = go.Scatter3d(x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color='black', width=0.3),
                        hoverinfo='none')

#create a trace for the nodes
#color=community_label,text=club_labels,
trace_nodes = go.Scatter3d(x=x_nodes,
                         y=y_nodes,
                        z=z_nodes,
                        mode='markers',
                        marker=dict(symbol='circle',
                                    size=6,
                                    color=community_label, #color the nodes according to their community
                                    colorscale='spectral',
                       line=dict(color='black', width=0.2)),
                        text=names)

#we need to set the axis for the plot 
axis = dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=True,
            showticklabels=False,
            title='')


#also need to create the layout for our plot
layout = go.Layout(title="Network, Cutoff = " + str(CUTOFF),
                width=1000,
                height=1000,
                showlegend=False,
                scene=dict(xaxis=dict(axis),
                        yaxis=dict(axis),
                        zaxis=dict(axis),
                        ),
                margin=dict(t=100),
                hovermode='closest')

#Include the traces we want to plot and create a figure
data = [trace_edges, trace_nodes]
fig = go.Figure(data=data, layout=layout)

fig.show()
import plotly.io as pio
pio.write_html(fig, auto_open=True, file = str(CUTOFF)+"_traj.html")
