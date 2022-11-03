
import shutil, os, json

def main():
    outdir_cpet = "/ocean/projects/che160019p/santi92/cpet/"
    # get subdirectories in folder
    subdirs = [x[0] for x in os.walk(outdir_cpet)]
    # create new folder for communities
    outdir_communities = "/ocean/projects/che160019p/santi92/communities/"
   

    for i in subdirs: 
        prot_id = i.split("/")[-1]
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


main()


# delete any file with name of 4 characters
"""import pandas as pd 
names = []
file_names = "../../data/communities/topo_file_list.txt"
dist_mat_name = "../../data/communities/distance_matrix.dat"

with open(file_names, "r") as f:
    for line in f:
        file_name = line.split("/")[-1]
        names.append(file_name.split("_")[3]+"_"+file_name.split("_")[1])

data = pd.read_csv(dist_mat_name, sep = " ", header = None)
data = data.iloc[: , :-1]
data.columns = names """