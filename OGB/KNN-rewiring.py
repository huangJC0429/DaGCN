import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
import networkx as nx
from sklearn.cluster import KMeans
import os
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
# from utils import get_dataset_split
import os.path as osp
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
# import cvae_pretrain
import sys


class Base:

    def __init__(self, adj, features, device):
        self.adj = adj
        self.features = features.to(device)
        self.device = device
        self.cached_adj_norm = None

    def get_adj_norm(self):
        if self.cached_adj_norm is None:
            adj_norm = preprocess_adj(self.adj, self.device)
            self.cached_adj_norm= adj_norm
        return self.cached_adj_norm

    def make_loss(self, embeddings):
        return 0

    def transform_data(self):
        return self.get_adj_norm(), self.features

class KNN_Rewiring(Base):

    def __init__(self, adj, features, nhid, device, idx_train, args):
        #  self.adj = adj
        self.args = args

        # self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        self.labeled = idx_train.cpu().numpy()
        # self.all = np.arange(adj.shape[0])
        # self.unlabeled = np.array([n for n in self.all if n not in idx_train])
        self.nclass = 1
        self.pseudo_labels = None
        self.A_feat = None

        # degree = self.adj.sum(0).A1
        k = args.k
        if not os.path.exists(f'./knn_graph/{args.dataset}_sims_{k}.npz'):
            from sklearn.metrics.pairwise import cosine_similarity
            # from sklearn.metrics import pairwise_distances
            # from scipy.spatial.distance import jaccard
            # metric = jaccard
            metric = "cosine"
            # sims = pairwise_distances(self.features, self.features, metric=metric)
            # features[features!=0] = 1
            print(features.shape)
            # features = sp.csr_matrix(features)
            #
            # sims = cosine_similarity(features, dense_output=False)
            # print(sims)
            # exit()
            # sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
            # for i in range(len(sims)):
            #     indices_argsort = np.argsort(sims[i])
            #     sims[i, indices_argsort[: -k]] = 0
            nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(features)
            distances, indices = nbrs.kneighbors(features)
            n_samples = len(features)
            adjacency_matrix = lil_matrix((n_samples, n_samples), dtype=int)
            for i in range(n_samples):
                adjacency_matrix[i, indices[i]] = 1
            self.A_feat = adjacency_matrix.tocsr()
            # print(indices)
            # exit()
            #
            # self.A_feat = sp.csr_matrix(sims)
            sp.save_npz(f'./knn_graph/{args.dataset}_sims_{k}.npz', self.A_feat)
            print("complete save!")
        else:

            print(f'loading {args.dataset}_sims_{k}.npz')
            self.A_feat = sp.load_npz(f'./knn_graph/{args.dataset}_sims_{k}.npz')


    def transform_data(self, lambda_=None):


        if self.cached_adj_norm is None:
            r_adj = self.A_feat
            r_adj = preprocess_adj(r_adj, self.device)
            self.cached_adj_norm = r_adj

            # self.features[self.features!=0] = 1

        return self.cached_adj_norm# , self.features

    def make_loss(self, embeddings):
        return 0

def preprocess_features(features, device):
    return features.to(device)

def preprocess_adj(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = aug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def preprocess_adj_noloop(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = noaug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def noaug_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

# from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor, get_dataset, row_l1_normalize, GCN_norm
from tqdm import trange
import sys
import argparse
import random
from torch_sparse import SparseTensor
from torch_sparse import sum as sparsesum
exc_path = sys.path[0]
from torch_geometric.nn.conv.gcn_conv import gcn_norm

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=1)
parser.add_argument("--concat", type=int, default=1)
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')

parser.add_argument('--dataset', default='arxiv', help='Dataset string.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--k', type=int, default=50, help=' K for knn.')


args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()
path = "data"
# Load data
dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)
split_idx = dataset.get_idx_split()
idx_train = split_idx["train"].to(device)

features = data.x



exc_path = sys.path[0]
cvae_model = torch.load("{}/model/{}.pkl".format(exc_path, args.dataset))
def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features.clone().detach(), dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        # augmented_features = cvae_pretrain.feature_tensor_normalize(augmented_features).detach()
        if args.cuda:
            X_list.append(augmented_features.to(device))
        else:
            X_list.append(augmented_features)
    return X_list

X_aug = get_augmented_features(args.concat)[0]
X_aug = X_aug.cpu().detach().numpy()
features = features.cpu().detach().numpy()
print(X_aug)
print(features)

aug_features = np.multiply(features, X_aug)# np.hstack((features, X_aug))

knn = KNN_Rewiring(None, aug_features, 8, device, idx_train, args)
# A = knn.transform_data()
# print(knn.A_feat)
# print(A)
# print(knn.A_feat)

# row, col = edge_index
# num_nodes = data.x.shape[0]
# adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
# norm_adj = gcn_norm(adj, add_self_loops=True).to_torch_sparse_coo_tensor()
# print(norm_adj)

# A = 0.1*A + 0.9*norm_adj
coo = knn.A_feat.tocoo()
indices = torch.LongTensor([coo.row, coo.col])

num_nodes = features.shape[0]
sp_A = SparseTensor(row=indices[0], col=indices[1], sparse_sizes=(num_nodes, num_nodes))
# print(sp_A)
# exit()
# print(sparsesum(sp_A, dim=0))
print(sp_A)
torch.save(sp_A, f'./knn_graph/{args.dataset}_sims_{args.k}.pt') # save as torch_sparse tensor.

