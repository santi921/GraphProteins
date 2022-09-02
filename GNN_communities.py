"""
This example implements the experiments for node clustering on citation networks
from the paper:
Mincut pooling in Graph Neural Networks (https://arxiv.org/abs/1907.00481)
Filippo Maria Bianchi, Daniele Grattarola, Cesare Alippi
"""
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.metrics.cluster import (
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
import community as community_louvain

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tqdm import tqdm

from spektral.datasets.citation import Cora
from spektral.layers.convolutional import GCSConv
from spektral.layers.pooling import MinCutPool
from spektral.utils.convolution import normalized_adjacency

