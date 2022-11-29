import shutil, os, json
from glob import glob
import pandas as pd
import numpy as np
import networkx as nx

def separate_md_runs(outdir_cpet, outdir_communities, split_flag, prot_id_ind):
    """
       Separates the cpet outs into different folders by protein
       Takes: 
            outdir_cpet: the directory where the cpet outs are
            outdir_communities: the directory where the communities will be
            split_flag: the flag to split the file name by
            prot_id_ind: the index of the protein id in the file name after split
    """
    # get subdirectories in folder
    subdirs = [x[0] for x in os.walk(outdir_cpet)]
    # create new folder for communities
    for i in subdirs: 
        prot_id = i.split(split_flag)[prot_id_ind]
        # check if there is a file contraining compressed
        if "compressed_dictionary.json" in os.listdir(i):
            #create new folder with protein id
            #os.mkdir(outdir_communities + prot_id)
            #copy compressed file to new folder
            shutil.copy(i+"/compressed_dictionary.json", outdir_communities + prot_id + "_compressed.json")
            compressed = json.load(open(i+"/compressed_dictionary.json"))
            for k,v in compressed.items():
                representative_proteins = v["name_center"]
                # move representative proteins to new folder
                shutil.copy(representative_proteins, outdir_communities)



def parse_compressed_directory(root = "../../md_traj/", distance_file = "md_cys.csv", cutoff = 7.5, index_ind=1, protein_ind=3):
    """
        Parses the compressed directory and returns a graph, names, counts, and protein_names_and_count
        Takes:
            root: the root directory of compressed files
            distance_file: the distance file in root
            cutoff: the cutoff distance for the graph
            index_ind: the index of the index in the distance filename
            protein_ind: the index of the protein name in the distance filename
        
        Returns:
            G: the graph
            names: the names of the proteins/nodes
            counts: the counts of the proteins/nodes
            protein_names_and_count: the names and counts of the proteins/nodes for plotting
    """
    counts = []     
    
    distance_file = root + distance_file
    name_file = root + "topo_file_list.txt"
    #compressed_dictionaries = glob(root + "*compressed.json")
    #dist_matrix = pd.read_csv(distance_file, index_col=0).to_numpy()
    dist_matrix = pd.read_csv(distance_file, delimiter = ",").to_numpy()
    #names = np.genfromtxt(dist_mat, delimiter=',', dtype=str, max_rows=1)
    
    names = np.genfromtxt(name_file, dtype=str)
    
    std = np.std(dist_matrix.flatten())
    dist_matrix = (dist_matrix) / std
    dist_mask = np.where(dist_matrix > cutoff, 0, dist_matrix)
    G = nx.from_numpy_array(dist_mask)
    
    names_stripped = [i.split("/")[-1] for i in list(names)]
    
    index_stripped = [i.split("_")[index_ind] for i in names_stripped]
    protein_names = [i.split("_")[protein_ind] for i in names_stripped]
    protein_names_and_count = []

    for ind, i in enumerate(protein_names):
        with open(root + i + "_compressed.json", "r") as f:
            compressed_dict = json.load(f)
        for k, v in compressed_dict.items():
            ind_center = int(v["name_center"].split("/")[-1].split("_")[index_ind])
            if(ind_center == int(index_stripped[ind])):
                count_temp = v["count"]    
                counts.append(int(count_temp))
                protein_names_and_count.append(i + "_" + str(count_temp) + "_ind_" + str(ind_center))
    
                break
        else: 
            print("warning!, no match for ", i)

    return G, names, counts, protein_names_and_count