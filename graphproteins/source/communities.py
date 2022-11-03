
from cdlib.algorithms.attribute_clustering import ilouvain

import pandas as pd 
from cdlib import algorithms

from graphproteins.utils.data import Protein_Dataset_Single
from graphproteins.utils.models import *
from graphproteins.utils.communities import *
from graphproteins.utils.plot import plot_graph_w_communities


def communities(target_file = "distance_matrix_0930.csv", cutoff=0.2, c = 1): 
    
    activity = pd.read_csv("../../data/protein_data.csv")
    names = activity["name"]
    selectivity = activity["label"].tolist()
    
    dataset = Protein_Dataset_Single(
        name = 'prot datast',
        labels=selectivity,
        url="../../data/" + target_file,
        raw_dir="../../data/",
        save_dir="../../data/",
        force_reload=True,
        verbose=True,
        cutoff = cutoff,
        header = True,
        c = c
    )
    #distances_as_weights = dataset.graph_nx
    
    labels_as_dict = {}
    for ind, node in enumerate(dataset.graph_nx.nodes()):
        labels_as_dict[node]={"l1":dataset.labels[ind]}
    
    leiden_coms = algorithms.leiden(dataset.graph_nx)
    print("Mod Score: " + str(leiden_coms.newman_girvan_modularity().score))

    community_algo = leiden_coms
    community_label = [0 for i in community_algo.to_node_community_map()]
    ind = 0 
    for k,v in dict(community_algo.to_node_community_map()).items():
        community_label[ind] = v[0]
        ind+=1

    names = list(dict(community_algo.to_node_community_map()).keys())
    names_w_com_label = []
    for k,v in dict(community_algo.to_node_community_map()).items():
        names_w_com_label.append(str(k)+"_"+str(v[0]))

    # label with activity type 
    activity_label = []
    for i in names:     
        activity = labels_as_dict[i]["l1"]
        # C, H, Y normally
        if(activity == 0): color = "magenta"
        elif(activity== 1): color = "green"
        else: color = "skyblue"
        activity_label.append(color)    
    
    plot_graph_w_communities(
        dataset, 
        names_w_com_label, 
        community_label, 
        cutoff, 
        save=True)
    
    plot_graph_w_communities(
        dataset, 
        names, 
        activity_label, 
        cutoff, 
        save=True)

    activity_com_dict, com_dict = {}, {}

    for k,v in dict(community_algo.to_node_community_map()).items():
        if(str(v[0]) in com_dict.keys()): 
            com_dict[str(v[0])].append(k) 
        else: com_dict[str(v[0])] = [k]

    for k,v in dict(community_algo.to_node_community_map()).items():
        if(str(v[0]) in activity_com_dict.keys()): 
            activity_com_dict[str(v[0])].append(str(labels_as_dict[k]["l1"])) 
        else: activity_com_dict[str(v[0])] = [str(labels_as_dict[k]["l1"])]
        
    for k, v in activity_com_dict.items():
        community_breakdown(v)
    
communities(target_file = "distance_matrix_0930_filtered.csv", 
            cutoff = 0.19, 
            c = 0.1)

