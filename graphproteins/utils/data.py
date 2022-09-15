import os
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
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):

        super(Protein_Dataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)


    def pull(self):
        # path to store the file
        file_path = os.path.join(self.raw_dir, self.name)
        self.file_path = file_path


    def process(self):
        # Skip some processing code
        # === data processing skipped ===
        
        file = self.url
        #'../data/distance_matrix_trajectories_only_skip3.csv' # 6 classes
        dist_matrix = np.genfromtxt(file, delimiter=',', skip_header=1)
        header = np.genfromtxt(file, delimiter=',', dtype=str, max_rows=1)
        #mean = np.mean(dist_matrix)
        std = np.std(dist_matrix)
        dist_matrix = (dist_matrix ) / std
        dist_mask = np.where(dist_matrix > 0.55, 0, dist_matrix)
        G = nx.from_numpy_array(dist_mask)
        unique_prots = [i.split("_")[0] for i in header]
        unique_prots = list(set(unique_prots))
        one_hot_list = [unique_prots.index(i.split("_")[0]) for i in header]
        mapping =  { x:ind for  x, ind in enumerate(header) }
        mapping_one_hot =  { x:ind for  x, ind in enumerate(one_hot_list) }
        print(G)
        G_relab = nx.relabel_nodes(G, mapping)


        self.labels = list(mapping_one_hot.values())
        self.graph_nx = G_relab
        self.graph = dgl.from_networkx(G)

        
        X_train, idx_test = train_test_split(list(range(len(self.labels))), test_size=0.2, random_state=1)
        idx_train, idx_val  = train_test_split(X_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
        #print(len(self.labels))
        train_mask = _sample_mask(idx_train, int(len(self.labels)))
        val_mask = _sample_mask(idx_val, int(len(self.labels)))
        test_mask = _sample_mask(idx_test, int(len(self.labels)))
    

        # splitting masks
        self.graph.ndata['train_mask'] = generate_mask_tensor(np.array(train_mask))
        self.graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
        self.graph.ndata['test_mask'] = generate_mask_tensor(test_mask)
        # node labels
        self.graph.ndata['label'] = torch.tensor(self.labels)
        self._num_classes = int(len(unique_prots))

        '''
        # build graph
        g = dgl.graph(graph)
        # splitting masks
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
        # node labels
        g.ndata['label'] = torch.tensor(labels)
        # node features
        #g.ndata['feat'] = torch.tensor(_preprocess_features(features),
        #                               dtype=F.data_type_dict['float32'])
        self._num_labels = onehot_labels.shape[1]
        self._labels = labels
        # reorder graph to obtain better locality.
        self._g = dgl.reorder_graph(g)
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