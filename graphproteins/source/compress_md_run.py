import json
from multiprocessing import Process, Manager
from rmsd import rmsd, kabsch_rmsd, quaternion_rmsd, get_coordinates_pdb
from os import listdir
from glob import glob
from time import time
import numpy as np

def get_rmsd(A, B):
    try:
        _, A_coord = get_coordinates_pdb(A)
        _, B_coord = get_coordinates_pdb(B)
        t1= time()
        rmsd_val = rmsd(A_coord, B_coord)
        t2=time()
        time_rmsd = t2-t1

        t1= time()
        rmsd_kab = kabsch_rmsd(A_coord, B_coord)
        t2= time()
        time_kab = t2-t1

        #print(time_rmsd, time_kab)
        return np.min([rmsd_val, rmsd_kab])
    except: return 10

def get_rmsd_one_is_coord(A, B_coord):
    try: 
        _, A_coord = get_coordinates_pdb(A)
        #_, B_coord = rmsd.get_coordinates_pdb(B)
        t1= time()
        rmsd_val = rmsd(A_coord, B_coord)
        t2=time()
        time_rmsd = t2-t1

        t1= time()
        rmsd_kab = kabsch_rmsd(A_coord, B_coord)
        t2= time()
        time_kab = t2-t1

        #print(time_rmsd, time_kab)
        return np.min([rmsd_val, rmsd_kab])
    except: return 10

def process_frames(frames, dict_compressed, dict_pdb_xyz, id_prot, cutoff = 1.2):

    id_0 = frames[0]
    id_0 = id_0.split("/")[-1].split("_")[0]
    _, coord = get_coordinates_pdb(frames[0])
    
    dict_compressed_temp = {str(id_0):1}
    dict_pdb_xyz_compress_temp = {id_0: coord}

    for ind, j in enumerate(frames): 
        for key, xyz in dict_pdb_xyz_compress_temp.items():
            rmsd_ret = get_rmsd_one_is_coord(j, xyz)
            if rmsd_ret < cutoff: 
                
                dict_compressed_temp[key] = dict_compressed_temp[key]+1
            else: 
                id_frame = j.split("/")[-1].split("_")[0]
                dict_compressed_temp[id_frame] = 1

    dict_compressed[id_prot] = dict_compressed_temp
    dict_pdb_xyz[id_prot] = dict_pdb_xyz_compress_temp

def main():
    folder = "/ocean/projects/che160019p/santi92/heme_traj/"
    
    manager = Manager()


    try: 
        with open("mappings.json") as outfile:
            dict_total_compressed_json = json.load(outfile)
            dict_total_compressed = manager.dict()
            for k, v in dict_total_compressed_json.items():
                dict_total_compressed[k] = v

        with open("xyz.json", "w") as outfile:
            dict_pdb_xyz_compress_json = json.load(outfile)
            dict_pdb_xyz_compress = manager.dict()
            for k, v in dict_pdb_xyz_compress_json.items():
                dict_pdb_xyz_compress[k] = v
    
    except: 
        dict_total_compressed, dict_pdb_xyz_compress = {}, {}

    total_files = glob(folder + "*.pdb")
    total_files_no_folder = [i.split("/")[-1] for i in total_files]
    protein_ids = [i.split("_")[2] for i in total_files_no_folder]
    protein_ids = list(set(protein_ids))
    
    print(protein_ids)
    jobs = []
    
    for i in protein_ids:
        
        frames = glob(folder + "*" + i + "*")
        # get first & store
        proteins_done = list(dict_total_compressed.keys())

        if i not in proteins_done: 
            print("processing protein {}".format(i))


            process_frames(frames, dict_total_compressed, dict_pdb_xyz_compress, i)
            with open("mappings.json", "w") as outfile:
                json.dump(dict_total_compressed, outfile, indent=4)
            with open("xyz.json", "w") as outfile:
                json.dump(dict_pdb_xyz_compress, outfile, indent=4)        

    for proc in jobs:
        proc.join()

main()