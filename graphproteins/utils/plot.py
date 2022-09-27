import networkx as nx
from cdlib import algorithms, evaluation
import plotly.graph_objects as go
import plotly.io as pio


def plot_graph_w_communities(dataset, names, community_label, cutoff, save = False): 
    x_edges, y_edges, z_edges=[], [], []

    spring_3D = nx.spring_layout(dataset.graph_nx_relab, dim=3, seed = 1)

    x_nodes = [spring_3D[i][0] for i in dataset.graph_nx_relab] # x-coordinates of nodes
    y_nodes = [spring_3D[i][1] for i in dataset.graph_nx_relab] # y-coordinates
    z_nodes = [spring_3D[i][2] for i in dataset.graph_nx_relab] # z-coordinates
    edge_list = dataset.graph_nx_relab.edges()

    #need to fill these with all of the coordiates
    for edge in edge_list:
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
                            text=names)

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
    
    if(save):
        pio.write_html(fig, auto_open=True, file = "../../data/"+str(cutoff)+"_traj.html")
