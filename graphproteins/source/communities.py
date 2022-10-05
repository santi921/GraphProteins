
from cdlib.algorithms.attribute_clustering import ilouvain
import numpy as np
from calendar import c
import pandas as pd 
import community as community_louvain
from cdlib import algorithms, evaluation
from cdlib.algorithms import eva

from graphproteins.utils.data import Protein_Dataset_MD, Protein_Dataset_Single
from graphproteins.utils.models import *
from graphproteins.utils.communities import *
from graphproteins.utils.plot import plot_graph_w_communities


def communities(target_file = "distance_matrix_0930.csv", cutoff=0.2): 
    
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
        header = True
    )
    #distances_as_weights = dataset.graph_nx
    
    labels_as_dict = {}
    for ind, node in enumerate(dataset.graph_nx.nodes()):
        labels_as_dict[node]={"l1":dataset.labels[ind]}

    #for u, v, weight in dataset.graph_nx.edges.data("weight"):
    #    if weight is not None:
    #        dataset.graph_nx.edges[u, v]["weight"] = 1/weight
            #dataset.graph_nx.edges[u][v]["weight"] = 1/weight

    #for i in [0.8]:
    #    louvain_coms = algorithms.louvain(dataset.graph_nx, weight='weight', resolution=2.0)
    #    print("Mod Score: " + str(louvain_coms.newman_girvan_modularity()))
    #print("-"*20)
    leiden_coms = algorithms.leiden(dataset.graph_nx)
    #walkscan_coms = algorithms.walkscan(dataset.graph_nx, nb_steps= 4,eps= 0.8, min_samples= 5)
    #frc_fgsn_coms = algorithms.frc_fgsn(dataset.graph_nx, theta=1, eps=0.95, r=5)
    #ilouvain_coms = algorithms.ilouvain(dataset.graph_nx, labels_as_dict)
    #cpm_coms = algorithms.cpm(dataset.graph_nx)
    #chinese_coms = algorithms.chinesewhispers(dataset.graph_nx)
    #greedy_coms = algorithms.greedy_modularity(dataset.graph_nx)
    #core_coms = algorithms.core_expansion(dataset.graph_nx)
    
    partition = community_louvain.best_partition(dataset.graph_nx)
    print("Mod Score: " + str(leiden_coms.newman_girvan_modularity().score))
    '''
    
    print("Mod Score: " + str(walkscan_coms.newman_girvan_modularity().score))
    print("Mod Score: " + str(frc_fgsn_coms.newman_girvan_modularity().score))
    print("Mod Score: " + str(ilouvain_coms.newman_girvan_modularity().score))
    print("Mod Score: " + str(cpm_coms.newman_girvan_modularity().score))
    print("Mod Score: " + str(chinese_coms.newman_girvan_modularity().score))
    print("Mod Score: " + str(greedy_coms.newman_girvan_modularity().score))
    print("Mod Score: " + str(core_coms.newman_girvan_modularity().score))
    '''
    #label with community labels
    #community_label = [partition[i] for i in partition]
    #louvain_coms.to_node_community_map()
    community_algo = leiden_coms

    community_label = [0 for i in community_algo.to_node_community_map()]
    ind = 0 
    for k,v in dict(community_algo.to_node_community_map()).items():
        community_label[ind] = v[0]
        ind+=1
    names = [i for i in partition]
    
    # label with activity type 
    activity_label = []
    for i in names:     
        ind_act = names.index(i)
        activity = selectivity[ind_act]
        if(activity == 'Y'): color = "magenta"
        elif(activity=='C'): color = "green"
        else: color = "skyblue"
        activity_label.append(color)    
    plot_graph_w_communities(
        dataset, 
        names, 
        community_label, 
        cutoff, 
        save=True)

    plot_graph_w_communities(
        dataset, 
        names, 
        activity_label, 
        cutoff, 
        save=True)

    com_dict = {}
    activity_com_dict = {}
    #for i in partition:
    #    if(partition[i] in com_dict.keys()): com_dict[partition[i]].append(i) 
    #    else: com_dict[partition[i]] = [i] 
    #for k, v in com_dict.items():
    #    community_breakdown(v)
    
    #community_label = [0 for i in community_algo.to_node_community_map()]
    ind = 0 
    print(community_algo.to_node_community_map())
    for k,v in dict(community_algo.to_node_community_map()).items():
        if(str(v[0]) in com_dict.keys()): 
            com_dict[str(v[0])].append(k) 
        else: com_dict[str(v[0])] = [k]

    for k,v in dict(community_algo.to_node_community_map()).items():
        if(str(v[0]) in activity_com_dict.keys()): 
            activity_com_dict[str(v[0])].append(str(labels_as_dict[k]["l1"])) 
        else: activity_com_dict[str(v[0])] = [str(labels_as_dict[k]["l1"])]
        
    #print(com_dict)
    #for k, v in com_dict.items():
    #    community_breakdown(v)
    for k, v in activity_com_dict.items():
        community_breakdown(v)
    #names = [i for i in partition]

communities(target_file = "distance_matrix_0930_filtered.csv", 
            cutoff = 0.18)

#file = 'distance_matrix_full_skip3.csv'
#file = 'distance_matrix_trajectories_only_skip3.csv'
#file = 'distance_matrix.csv'
#file = 'distance_matrix_chi3.csv'
#file = 'distance_matrix_with_originals.csv'