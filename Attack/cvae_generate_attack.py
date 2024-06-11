import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import cvae_pretrain

from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor, get_dataset, row_l1_normalize
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse as sp
from gcn.models import GCN
from tqdm import trange

exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument("--pretrain_lr", type=float, default=0.01)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')
parser.add_argument('--ptb_rate', type=float, default=0.0, help='pertubation rate')
parser.add_argument('--attack', type=str, default='meta', choices=['no', 'meta', 'random', 'nettack'])

parser.add_argument('--dataset', default='pubmed',
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

args = parser.parse_args()


# # torch.save(torch.tensor([0]), "./model/%s_"+str(args.ptb_rate)+".pkl" % args.dataset)
# torch.save(torch.tensor([0]), "./model/"+args.dataset+"_"+str(args.ptb_rate)+".pkl")
# exit()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

# Load data

data = np.load("./data/save_data/"+args.dataset+".npy", allow_pickle=True).item()
# adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
adj, features, labels, idx_train, idx_val, idx_test = data['adj'], data['x'], data['y'], data['idx_train'], data['idx_val'], data['idx_test']
adj = sp.csr_matrix(adj)
features = features.A

if args.ptb_rate == 0:
    perturbed_adj = adj
else:
    if args.attack == 'meta':
        if args.ptb_rate == 0.1 or args.ptb_rate == 0.2:
            perturbed_data_file = "./meta_modified_graph_data/%s_meta_adj_%.1f.npz" % (args.dataset, args.ptb_rate)
        else:
            perturbed_data_file = "./meta_modified_graph_data/%s_meta_adj_%.2f.npz" % (args.dataset, args.ptb_rate)

    if args.attack == 'random':
        perturbed_data_file = "./random_attack_data/%s_random_%.1f_adj.npz" % (
            args.dataset, args.ptb_rate)

    print("perturbed data file is:", perturbed_data_file)
    perturbed_adj = sp.load_npz(perturbed_data_file)

adj = perturbed_adj
# Normalize adj and features
adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
features_normalized = normalize_features(features)

# To PyTorch Tensor
labels = torch.LongTensor(labels)
# labels = torch.max(labels, dim=1)[1]
features_normalized = torch.FloatTensor(features_normalized)
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
idx_train = torch.LongTensor(idx_train)

_, cvae_model = cvae_pretrain.generated_generator(args, device, adj, features, labels, features_normalized,
                                                  adj_normalized, idx_train, edge_index=None)
#  We have supplied a pretrained model named "%s.pkl" for the %s dataset. If you wish to use your own pretrained model, please save it with the filename "model/%s_.pkl" to avoid overwriting our provided models.
# torch.save(cvae_model, "./model/%s_"+str(args.ptb_rate)+".pkl" % args.dataset)
torch.save(cvae_model, "./model/"+args.dataset+"_"+args.attack+str(args.ptb_rate)+".pkl")