import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import cvae_pretrain
import cvae_pretrain_large

from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor, get_dataset, row_l1_normalize, get_dataset_split
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse as sp
from gcn.models import GCN
from tqdm import trange

exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_epochs", type=int, default=10)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')

parser.add_argument('--dataset', default='CS',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument("--latent_size", type=int, default=3)
parser.add_argument("--pretrain_lr", type=float, default=0.01)
parser.add_argument("--total_iterations", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=8192)

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

# Load data
if args.dataset in ['cora', 'citeseer', 'pubmed']:
    adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
    path = "data"
    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=path,
    )
    data = dataset._data.to(device)

    # Normalize adj and features
    features = features.toarray()
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    features_normalized = normalize_features(features)

    # To PyTorch Tensor
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    features_normalized = torch.FloatTensor(features_normalized)
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    idx_train = torch.LongTensor(idx_train)

    _, cvae_model = cvae_pretrain.generated_generator(args, device, adj, features, labels, features_normalized,
                                                      adj_normalized, idx_train, edge_index=data.edge_index, model='gcn')
    #  We have supplied a pretrained model named "%s.pkl" for the %s dataset. If you wish to use your own pretrained model, please save it with the filename "model/%s_.pkl" to avoid overwriting our provided models.
    torch.save(cvae_model, "model/%s_.pkl" % args.dataset)

else:
    path = "data"
    # Load data
    # adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=path,
    )
    data = dataset._data.to(device)

    edge_index, features,  labels = data.edge_index, data.x, data.y.long().squeeze()
    train_mask, val_mask, test_mask = get_dataset_split(args.dataset, data, args.dataset, 0)
    idx_train, idx_val, idx_test = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    row, col = edge_index
    num_nodes = features.shape[0]
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    adj_normalized = gcn_norm(adj.to_symmetric(), add_self_loops=True)#
    # features_normalized = row_l1_normalize(features)
    adj = sp.csr_matrix(adj.to_dense().cpu().detach().numpy())
    features = features#.cpu().detach().numpy()

    cvae_model = cvae_pretrain_large.generated_generator(args, device, adj.tocsr(), features)
    # _, cvae_model = cvae_pretrain.generated_generator(args, device, adj, features, labels, features_normalized,
    #                                                   adj_normalized, idx_train, edge_index=data.edge_index)
    #  We have supplied a pretrained model named "%s.pkl" for the %s dataset. If you wish to use your own pretrained model, please save it with the filename "model/%s_.pkl" to avoid overwriting our provided models.
    torch.save(cvae_model, "model/%s_.pkl"%args.dataset)
