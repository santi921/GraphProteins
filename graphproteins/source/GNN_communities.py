import torch
import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dgl.data.utils import generate_mask_tensor
from graphproteins.utils.dataset import Protein_Dataset
from graphproteins.utils.models import *
from dgl.nn import EdgeWeightNorm, GraphConv


dataset = Protein_Dataset(
    name = 'prot datast',
    url="../../data/distance_mats/distance_matrix_trajectories_only_skip3.csv",
    raw_dir="../../data/",
    save_dir="../../data/",
    force_reload=True,
    verbose=True,
    cutoff = 1.2
)


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feats']
    labels = g.ndata['label']
    edge_weight = g.edata['feats']
    norm = EdgeWeightNorm(norm='right', eps=0.001)
    print(edge_weight.shape)
    print(edge_weight)
    norm_edge_weight = norm(g, edge_weight)

    #conv = GraphConv(10, 2, norm='none', weight=True, bias=True)
    #res = conv(g, feat, edge_weight=norm_edge_weight)
    
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    

    for e in range(5000):
        # Forward
        logits = model(g, features, norm_edge_weight)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))


node_features = torch.tensor(dataset.graph.ndata["feats"].clone().detach(), dtype=torch.float32)
node_labels = torch.tensor(dataset.graph.ndata['label'].clone().detach(), dtype=torch.uint8)
train_mask = dataset.graph.ndata['train_mask']
valid_mask = dataset.graph.ndata['val_mask']
test_mask = dataset.graph.ndata['test_mask']
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)

# Create the model with given dimensions
#dataset.graph.ndata['degree'].shape[1]
model = GCN(n_features, 16, n_labels)
#model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)

graph = dataset.graph.to('cuda')
train(graph, model.to('cuda'))
