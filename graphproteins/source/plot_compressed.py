import os, json

import numpy as np
import pandas as pd
import networkx as nx 
import plotly.graph_objects as go
import plotly.io as pio
from glob import glob
from graphproteins.utils.data import parse_compressed_directory

dict_actions = {
    "1apx": "H", 
    "1dgh": "Y",
    "1ebe": "H", 
    "1gwf": "Y",
    "1hch": "H", 
    "1u5u": "Y",
    "3abb": "C", 
    "3hb6": "Y", 
    "4g3j": "C" 
}




def main(): 

    label_color, size_map = [], []
    x_edges, y_edges, z_edges=[], [], []

    """
    cutoff = 4.0
    root = "../../data/md_traj_hemes/"
    distance_file = "distance_matrix.csv"
    index_ind = 0
    protein_ind = 1
    cystine = False
    """
    # cystine
    cutoff = 6.5
    root = "../../data/md_traj/"
    distance_file = "md_cys.csv"
    index_ind = 1
    protein_ind = 3
    cystine = True

    G, names, counts, protein_names_and_count  = parse_compressed_directory(
        root = root, 
        distance_file = distance_file,
        cutoff = cutoff,
        protein_ind=protein_ind,
        index_ind=index_ind)

    # color list accordin to protein name
    """

    """
    names = []
    if cystine:
        for i in range(len(protein_names_and_count)):
            protein_name = protein_names_and_count[i].split("_")[0]
            names.append(protein_name)
            #color = dict_prot_name[protein_name]
        names_set = list(set(names))
        # randomly assign random colors to each protein from list of colors
        for i in range(len(protein_names_and_count)):
            protein_name = protein_names_and_count[i].split("_")[0]
            color = names_set.index(protein_name)
            label_color.append(color)

    else: 
        for i in range(len(protein_names_and_count)):
            protein_name = protein_names_and_count[i].split("_")[0]
            activity = dict_actions[protein_name]
            if activity == "H":
                label_color.append("red")
            elif activity == "Y":
                label_color.append("green")
            elif activity == "C":
                label_color.append("blue")
            else:
                label_color.append("black")

    max_color_value, min_color_value = max(np.array(counts)), min(np.array(counts))
    for i in counts:
        size_map.append(20 * (i)/(max_color_value-min_color_value))
    if label_color == []:
        label_color = size_map
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
    trace_edges = go.Scatter3d(
                            x=x_edges,
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
    if cystine:
        

        trace_nodes = go.Scatter3d(x=x_nodes,
                            y=y_nodes,
                            z=z_nodes,
                            mode='markers',
                            marker=dict(symbol='circle',
                                        size=size_map,
                                        color = label_color,
                        line=dict(color='black', width=0.2)),
                            text=protein_names_and_count)
    else: 
        trace_nodes = go.Scatter3d(x=x_nodes,
                            y=y_nodes,
                            z=z_nodes,
                            mode='markers',
                            marker=dict(symbol='circle',
                                        size=size_map,
                                        colorscale="YlOrRd",
                                        color = label_color,
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
    if cystine:
            pio.write_html(fig, auto_open=False, file = "../../reporting/html/" + str(cutoff)+"_compressed_traj_cyst.html")
    else: 
        pio.write_html(fig, auto_open=False, file = "../../reporting/html/" + str(cutoff)+"_compressed_traj.html")
    
    fig.show()
    
main()