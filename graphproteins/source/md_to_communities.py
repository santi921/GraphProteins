
import networkx as nx 
import numpy as np
import pandas as pd
from cdlib import algorithms, evaluation

import os
def main():
    folder = "../../data/md_traj/"
    dist_mat = "../../data/md_traj/md_cys.csv"
    cutoff = 0.0
    #dist_matrix = np.genfromtxt(dist_mat, delimiter=',', skip_header=1)
    dist_matrix = pd.read_csv(dist_mat, index_col=0)
    #print(dist_matrix.columns)

    #std = np.std(dist_matrix)
    #dist_matrix = (dist_matrix) / std
    #dist_mask = np.where(dist_matrix > cutoff, 0, dist_matrix)
    
    G = nx.from_numpy_array(dist_matrix.to_numpy())
    nx.draw(G)
    #leiden_coms = algorithms.leiden(G)
    #print("Mod Score: " + str(leiden_coms.newman_girvan_modularity().score))

    
main()