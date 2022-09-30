import os
import scipy as sp
import dgl.backend as F
from typing import Generator 
import dgl
import torch
import numpy as np 
import networkx as nx
from copy import deepcopy
from dgl.data import DGLDataset
from dgl.data.utils import generate_mask_tensor
from sklearn.model_selection import train_test_split

class Protein_Dataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """


    def __init__(self, name, url=None, raw_dir=None, save_dir=None,
                 hash_key=(), force_reload=False, verbose=False, 
                 transform=None, cutoff = 0.55):
        self._name = name
        self._url = url
        self._force_reload = force_reload
        self._verbose = verbose
        self._hash_key = hash_key
        self._hash = self._get_hash()
        self._transform = transform
        self.cutoff = cutoff
        self._raw_dir = raw_dir
        self._save_dir = save_dir

        # if no dir is provided, the default dgl download dir is used.
        # if raw_dir is None:
        #     self._raw_dir = get_download_dir()
        # else:
        #     self._raw_dir = raw_dir
        if save_dir is None:
            self._save_dir = self._raw_dir
        else:
            self._save_dir = save_dir

        self._load()


    def pull(self):
        # path to store the file
        file_path = os.path.join(self.raw_dir, self.name)
        self.file_path = file_path


    def process(self):
        # Skip some processing code
        # === data processing skipped ===
        
        file = self.url
        cutoff = self.cutoff
        #'../data/distance_matrix_chi3.csv' # 6 classes
        
        dist_matrix = np.genfromtxt(file, delimiter=',', skip_header=1)
        header = np.genfromtxt(file, delimiter=',', dtype=str, max_rows=1)
        mean = np.mean(dist_matrix)
        std = np.std(dist_matrix)
        dist_matrix = (dist_matrix) / std
        dist_mask = np.where(dist_matrix > cutoff, 0, dist_matrix)
        G = nx.from_numpy_array(dist_mask)
        print(G)

        unique_prots = [i.split("_")[0] for i in header]
        unique_prots = list(set(unique_prots))
        one_hot_list = [unique_prots.index(i.split("_")[0]) for i in header]
        mapping =  { x:ind for  x, ind in enumerate(header) }
        mapping_one_hot =  { x:ind for  x, ind in enumerate(one_hot_list) }
        G_relab = nx.relabel_nodes(G, mapping)
        print(G_relab)
        
        self.labels = list(mapping_one_hot.values())
        self.graph_nx = G
        self.graph_nx_relab = G_relab
        
        # set graph degs
        degree_dict = {}
        between_dict = nx.betweenness_centrality(G)
        harmonic_dict = nx.harmonic_centrality(G)
        eigen_dict = nx.eigenvector_centrality(G, max_iter=1000)

        for (node, val) in G.degree(): degree_dict[node] = val 
        nx.set_node_attributes(G, degree_dict, 'degree')
        nx.set_node_attributes(G, between_dict, 'between')
        nx.set_node_attributes(G, harmonic_dict, 'harmonic')
        nx.set_node_attributes(G, eigen_dict, 'eigen')

        # set weights for edges
        self.graph = dgl.DGLGraph()
        self.graph = dgl.from_networkx(
            G.to_directed(), 
            node_attrs = ["degree", "between", 'harmonic', 'eigen'], 
            edge_attrs=["weight"])
        
        self.graph.ndata["feats"] = torch.vstack([
            self.graph.ndata['degree'],
            self.graph.ndata['between'],
            self.graph.ndata['harmonic'],
        ]).T
        
        self.graph.edata["feats"] = torch.tensor([1/i for i in self.graph.edata['weight']])
        
        
        X_train, idx_test = train_test_split(list(range(len(self.labels))), test_size=0.2, random_state=1)
        idx_train, idx_val  = train_test_split(X_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
        
        train_mask = _sample_mask(idx_train, int(len(self.labels)))
        val_mask = _sample_mask(idx_val, int(len(self.labels)))
        test_mask = _sample_mask(idx_test, int(len(self.labels)))
    
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.idx_val = idx_val
        
        # splitting masks
        self.graph.ndata['train_mask'] = generate_mask_tensor(np.array(train_mask))
        self.graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
        self.graph.ndata['test_mask'] = generate_mask_tensor(test_mask)

        # node labels
        self.graph.ndata['label'] = torch.tensor(self.labels)
        
        self._num_classes = int(len(unique_prots))
        self.graph = dgl.reorder_graph(self.graph)
        self.graph = dgl.add_self_loop(self.graph)
        self._g = self.graph

        '''
        #g.ndata['feat'] = torch.tensor(_preprocess_features(features),
        #                               dtype=F.data_type_dict['float32'])
        '''


    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g


    def __len__(self):
        return 1


    def save(self):
        pass


    #def save(self):
    #    """save the graph list and the labels"""
    #    graph_path = os.path.join(self.save_path,
    #                              self.save_name + '.bin')
    #    info_path = os.path.join(self.save_path,
    #                             self.save_name + '.pkl')
    #    save_graphs(str(graph_path), self._g)
    #    save_info(str(info_path), {'num_nodes': self.num_nodes,
    #                               'num_rels': self.num_rels})

    #def load(self):
    #    # load processed data from directory `self.save_path`
    #    graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    #    self.graphs, label_dict = load_graphs(graph_path)
    #    self.labels = label_dict['labels']
    #    info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    #    self.num_classes = load_info(info_path)['num_classes']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)



def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.asarray(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.asarray(list(map(classes_dict.get, labels)),
                               dtype=np.int32)
    return labels_onehot
