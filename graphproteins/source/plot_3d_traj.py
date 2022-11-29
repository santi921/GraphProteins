import networkx as nx
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cdlib import algorithms, evaluation
from networkx.algorithms.community.centrality import girvan_newman
import pandas as pd
import plotly.graph_objects as go


CUTOFF = 0.55
file = 'distance_matrix_trajectories_only_skip3.csv'
file = 'distance_matrix.csv'
#sfile = 'distance_matrix_with_originals.csv'

dist_matrix = np.genfromtxt(file, delimiter=',', skip_headser=1)
header = np.genfromtxt(file, delimiter=',', dtype=str, max_rows=1)
mean = np.mean(dist_matrix)
std = np.std(dist_matrix)
dist_matrix = (dist_matrix ) / std
#dist_mask = np.where(dist_matrix > 0.8, 0, dist_matrix)
dist_mask = np.where(dist_matrix > CUTOFF, 0, dist_matrix)
#dist_mask = 1 - dist_mask / np.max(dist_matrix)


frame = pd.DataFrame(np.array(dist_mask), columns=[i for i in header])
df_2rfb = frame[frame.columns.str.contains("2rfb_", case=False)]
df_3s79 = frame[frame.columns.str.contains("3s79_", case=False)]
df_4e2p = frame[frame.columns.str.contains("4e2p_", case=False)]
df_4ubs = frame[frame.columns.str.contains("4ubs_", case=False)]
df_6a17 = frame[frame.columns.str.contains("6a17_", case=False)]
community_label = []
[community_label.append(0) for i in range(np.count_nonzero(frame.columns.str.contains("2rfb_", case=False)))]
[community_label.append(1) for i in range(np.count_nonzero(frame.columns.str.contains("3s79_", case=False)))]
[community_label.append(2) for i in range(np.count_nonzero(frame.columns.str.contains("4e2p_", case=False)))]
[community_label.append(3) for i in range(np.count_nonzero(frame.columns.str.contains("4ubs_", case=False)))]
[community_label.append(4) for i in range(np.count_nonzero(frame.columns.str.contains("6a17_", case=False)))]

G = nx.from_numpy_array(dist_mask)
mapping =  { x:ind for  x, ind in enumerate(header) }

G_relab = nx.relabel_nodes(G, mapping)
print(G_relab)
pos = nx.spring_layout(G_relab)


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
                        text=header)

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

#fig.show()
import plotly.io as pio
pio.write_html(fig, auto_open=True, file = "../reporting/html/"+ str(CUTOFF) +".html")
