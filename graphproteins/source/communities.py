
from graphproteins.utils.communities import *
    
communities(target_file = "../../data/distance_mats/distance_matrix.csv", 
            label_file = "../../data/distance_mats/protein_data.csv",
            cutoff = 2, 
            c = 3)

