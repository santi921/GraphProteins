import os, json

import numpy as np
import pandas as pd
import networkx as nx 
import plotly.graph_objects as go
import plotly.io as pio
from glob import glob
from graphproteins.utils.data import parse_compressed_directory


def main(): 

    color_map_log, color_map = [], []
    x_edges, y_edges, z_edges=[], [], []

    # 
    cutoff = 2.6
    root = "../../data/md_traj_hemes/"
    distance_file = "distance_matrix.csv"
    index_ind = 0
    protein_ind = 1
    
    # cystine
    #cutoff = 7.5 
    #root = "../../data/md_traj/"
    #distance_file = "md_cys.csv"
    # index_ind = 1
    # protein_ind = 3

    G, names, counts, protein_names_and_count  = parse_compressed_directory(
        root = root, 
        distance_file = distance_file,
        cutoff = cutoff,
        protein_ind=protein_ind,
        index_ind=index_ind)

    max_color_value, min_color_value = max(np.array(counts)), min(np.array(counts))
    
    for i in counts:
        #color_map.append(np.log(i) * 100)
        color_map_log.append(25*i/max_color_value)
        color_map.append((i)/(max_color_value-min_color_value))
    
    spring_3D = nx.spring_layout(G,dim=3, seed = 1)
    
    x_nodes = [spring_3D[i][0] for i in G] # x-coordinates of nodes
    y_nodes = [spring_3D[i][1] for i in G] # y-coordinates
    z_nodes = [spring_3D[i][2] for i in G] # z-coordinates
    edge_list = G.edges()

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
    '''
    trace_nodes = go.Scatter3d(x=x_nodes,
                            y=y_nodes,
                            z=z_nodes,
                            mode='markers',
                            marker=dict(symbol='circle',
                                        size=6,
                                        colorscale='spectral',
                        line=dict(color='black', width=0.2)),
                            text=names)
    '''
    trace_nodes = go.Scatter3d(x=x_nodes,
                            y=y_nodes,
                            z=z_nodes,
                            mode='markers',
                            marker=dict(symbol='circle',
                                        size=color_map_log,
                                        colorscale="YlOrRd",
                                        color = color_map_log,
                        line=dict(color='black', width=0.2)),
                            text=protein_names_and_count)

    #we need to set the axis for the plot 
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=True,
                showticklabels=False,
                title='')


    #also need to create the layout for our plot
    layout = go.Layout(title="Network, Cutoff = " + str(cutoff),
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
    pio.write_html(fig, auto_open=False, file = "../../reporting/html/" + str(cutoff)+"_compressed_traj.html")
    fig.show()
    
main()