import pandas as pd 
from cdlib import algorithms
import numpy as np
from graphproteins.utils.dataset import Protein_Dataset_Single
from graphproteins.utils.models import *
from graphproteins.utils.plot import plot_graph_w_communities

def community_breakdown(coms_processed):
    unique_list = []
    count_list = []
    for i in coms_processed:
        if(not(i.split("_")[0] in unique_list)):
            unique_list.append(i.split("_")[0])
            count_list.append(0)
        ind = unique_list.index(i.split("_")[0])
        count_list[ind] += 1         
    
    sum_list = 0 
    for i in count_list: sum_list+=i 
    print("----------------------------------")
    print(sum_list)
    for ind, i in enumerate(unique_list):
        
        if i == '0': name = "Y"
        elif i == '1': name = "H"
        else: name = "C"

        print(name + "\t\t" + str(count_list[ind]/sum_list))


def communities(target_file = "../../data/distance_matrix.csv", label_file = "../../data/protein_data.csv", cutoff=0.2, c = 1): 
    activity = pd.read_csv(label_file)
    header = np.genfromtxt(target_file, delimiter=',', dtype=str, max_rows=1)
    
    activity = activity[activity["name"].isin(header)]
    names = activity["name"]
    selectivity = activity["label"].tolist()
    print(activity)

    dataset = Protein_Dataset_Single(
        name = 'prot datast',
        labels=selectivity,
        url= target_file,
        raw_dir="../../data/distance_mats/",
        save_dir="../../data/distance_mats/",
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