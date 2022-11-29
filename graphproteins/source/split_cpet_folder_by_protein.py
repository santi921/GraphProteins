
from graphproteins.utils.data import separate_md_runs

def main():

    split_flag = "/"
    prot_id_ind = -1
    outdir_cpet = "/ocean/projects/che160019p/santi92/cpet/"
    outdir_communities = "/ocean/projects/che160019p/santi92/communities/"
    separate_md_runs(outdir_cpet, outdir_communities, split_flag, prot_id_ind)

main()
