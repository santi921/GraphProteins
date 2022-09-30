
import numpy as np
from calendar import c

import community as community_louvain
from cdlib import algorithms

from graphproteins.utils.data import Protein_Dataset
from graphproteins.utils.models import *
from graphproteins.utils.communities import *
from graphproteins.utils.plot import plot_graph_w_communities


def communities(target_file = "distance_matrix_with_originals.csv", cutoff=1.2): 

    dataset = Protein_Dataset(
        name = 'prot datast',
        url="../../data/" + target_file,
        raw_dir="../../data/",
        save_dir="../../data/",
        force_reload=True,
        verbose=True,
        cutoff = cutoff
    )

    louvain_coms = algorithms.louvain(dataset.graph_nx, weight='weight', resolution=1.2)
    partition = community_louvain.best_partition(dataset.graph_nx_relab)
    print("Mod Score: " + str(community_louvain.modularity(partition, dataset.graph_nx_relab)))
    print("Mod Score: " + str(louvain_coms.newman_girvan_modularity()))

 
    #partition 
    community_label = [partition[i] for i in partition]
    louvain_coms.to_node_community_map()
    print(louvain_coms)
    community_label = [0 for i in louvain_coms.to_node_community_map()]
    for i in louvain_coms.to_node_community_map():
        community_label[i] = louvain_coms.to_node_community_map()[i][0]
    names = [i for i in partition]

    plot_graph_w_communities(dataset, names, community_label, cutoff)

    com_dict = {}
    for i in partition:
        if(partition[i] in com_dict.keys()): com_dict[partition[i]].append(i) 
        else: com_dict[partition[i]] = [i] 
    for k, v in com_dict.items():
        community_breakdown(v)


communities(target_file = "distance_matrix.csv", cutoff = 0.95)


#file = 'distance_matrix_full_skip3.csv'
#file = 'distance_matrix_trajectories_only_skip3.csv'
#file = 'distance_matrix.csv'
#file = 'distance_matrix_chi3.csv'
#file = 'distance_matrix_with_originals.csv'