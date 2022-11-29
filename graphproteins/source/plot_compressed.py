import os, json

import numpy as np
import pandas as pd
import networkx as nx 
import plotly.graph_objects as go
import plotly.io as pio
from glob import glob



def main(): 
    counts = [] 
    cutoff = 7.0
    root = "../../data/md_traj/"
    dist_mat = root + "md_cys.csv"
    name_file = root + "topo_file_list.txt"
    compressed_dictionaries = glob(root + "*compressed.json")
    
    dist_matrix = pd.read_csv(dist_mat, index_col=0).to_numpy()
    #names = np.genfromtxt(dist_mat, delimiter=',', dtype=str, max_rows=1)
    names = np.genfromtxt(name_file, dtype=str)

    std = np.std(dist_matrix.flatten())
    dist_matrix = (dist_matrix) / std
    dist_mask = np.where(dist_matrix > cutoff, 0, dist_matrix)
    G = nx.from_numpy_array(dist_mask)

    names_stripped = [i.split("/")[-1] for i in list(names)]
    index_stripped = [i.split("_")[1] for i in names_stripped]
    protein_names = [i.split("_")[3] for i in names_stripped]
    protein_names_and_count = []
    
    for ind, i in enumerate(protein_names):
        with open(root + i + "compressed.json", "r") as f:
            compressed_dict = json.load(f)
        for k, v in compressed_dict.items():
            index_center = v["index_center"]
            ind_center = int(v["name_center"].split("/")[-1].split("_")[1])
            if(ind_center == int(index_stripped[ind])):
                count_temp = v["count"]    
                counts.append(int(count_temp))
                protein_names_and_count.append(i + "_" + str(count_temp))
    
                break
        else: 
            print("warning!, no match for ", i)
    
    max_color_value = max(np.array(counts))
    min_color_value = min(np.array(counts))
    #color_scale = [[0, 'rgb(255,255,255)'], [1, 'rgb(0,0,0)']]
    color_map_log, color_map = [], []
    for i in counts:
        #color_map.append(np.log(i) * 100)
        color_map_log.append(10*i/max_color_value)
        color_map.append((i)/(max_color_value-min_color_value))
    red = [i * 255 for i in color_map]
    blue = [i * 255 for i in color_map]
    green = [i * 255 for i in color_map]

    spring_3D = nx.spring_layout(G,dim=3, seed = 1)
    #print(spring_3D)
    x_nodes = [spring_3D[i][0] for i in G]# x-coordinates of nodes
    y_nodes = [spring_3D[i][1] for i in G]# y-coordinates
    z_nodes = [spring_3D[i][2] for i in G]# z-coordinates
    edge_list = G.edges()


    x_edges, y_edges, z_edges=[], [], []

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
                                        colorscale="OrRd",
                                        color = color_map,
                                        #color=['rgb({},{},{})'.format(r, g, b) for r,g,b in zip(red, green, blue)],
                                        
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

    fig.show()
    #pio.write_html(fig, auto_open=True, file = str(cutoff)+"_traj.html")

main()