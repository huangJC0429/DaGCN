from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import copy
import random
import torch.nn.functional as F
import torch.optim as optim
import cvae_pretrain

from utils import load_data, accuracy, get_dataset, row_l1_normalize, GCN_norm, get_dataset_split
from gcn.models2 import DaGCN, GCN
from tqdm import trange

exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--concat", type=int, default=1)
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')

parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1.2, help='Lamda') # sample=1时，consis_loss相当于就sharp了一下
parser.add_argument('--I', type=bool, default=False, help='Lamda')
parser.add_argument('--k', type=int, default=20, help=' K for knn.')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

if args.I:
    args.I = 1.0
else:
    args.I = 0.0

# Adapted from GRAND: https://github.com/THUDM/GRAND
def consis_loss(logps, temp=args.tem):

    avg_p = torch.exp(logps)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = torch.mean((avg_p-sharp_p).pow(2).sum(1))
    return loss


def consis_loss2(out, out1, out2, temp=args.tem):
    logps = [out1, out2]
    ps = [torch.exp(p) for p in logps]
    # sum_p = 0.
    # for p in ps:
    #     sum_p = sum_p + p
    # avg_p = sum_p / len(ps)
    avg_p = out
    avg_p = torch.exp(avg_p)

    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = 0.5*loss / len(ps)
    return  loss


path = "data"
# Load data
# adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
dataset, evaluator = get_dataset(
    name=args.dataset,
    root_dir=path,
)
data = dataset._data.to(device)
adj, features, idx_train, idx_val, idx_test, labels = data.edge_index, data.x, data.train_mask, data.val_mask, data.test_mask, data.y
features_normalized = row_l1_normalize(features)
adj_normalized = GCN_norm(adj, features)

cvae_model = torch.load("{}/model/{}.pkl".format(exc_path, args.dataset))
knn_adj_norm = torch.load(f'./knn_graph/{args.dataset}_sims_{args.k}.pt')

def get_augmented_features():
    X_list = []
    cvae_features = torch.tensor(features.clone().detach(), dtype=torch.float32).to(device)

    z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
    augmented_features = cvae_model.inference(z, cvae_features)
    augmented_features = cvae_pretrain.feature_tensor_normalize(augmented_features).detach()
    if args.cuda:
        X_list.append(augmented_features.to(device))
    else:
        X_list.append(augmented_features)
    return X_list


if args.cuda:
    adj_normalized = adj_normalized.to(device)
    labels = labels.squeeze().long().to(device)
    features_normalized = features_normalized.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

all_val = []
all_test = []
all_unf = []
all_f = []

loss_trains = []
loss_vals = []
for i in trange(args.runs, desc='Run Train'):
    # Model and optimizer
    model = DaGCN(args=args,
                  nfeat=features.shape[1],
                  nhid=args.hidden,
                  nclass=labels.max().item() + 1,
                  dropout=args.dropout,
                  tau=args.tem)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.to(device)

    # Train model
    best = 999999999
    best_model = None
    best_X_list = None
    import time
    for epoch in range(args.epochs):
        s = time.time()
        model.train()
        optimizer.zero_grad()


        X_list1 = get_augmented_features()
        X_list2 = get_augmented_features()
        model_out, gnn_emb1, gnn_emb2 = model(X_list1 + [features_normalized], X_list2 + [features_normalized], adj_normalized, knn_adj_norm)

        output = torch.log_softmax(model_out, dim=-1)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])


        loss_consis = consis_loss(output)
        loss_consis2 = consis_loss2(model_out, gnn_emb1, gnn_emb2)

        loss_train = loss_train + args.lam * loss_consis + args.I*loss_consis2 # + 0.1*model.contrasive_loss(gnn_emb1, gnn_emb2)#  + 0.5*model.contrasive_loss(sharp_emb1, gnn_emb2)1.2

        loss_train.backward()
        optimizer.step()

        model.eval()

        e = time.time()
        # print("time:", e-s)

        # val_X_list = get_augmented_features(args.concat)
        output, _, _ = model(X_list1 + [features_normalized], X_list2 + [features_normalized], adj_normalized, knn_adj_norm)
        output = torch.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        test_acc = accuracy(output[idx_test], labels[idx_test])
        

        # print('Epoch: {:04d}'.format(epoch+1),
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'test_acc: {:.4f}'.format(test_acc.item()),)

        if loss_val < best:
            best = loss_val
            best_model = copy.deepcopy(model)
            # best_X_list = copy.deepcopy(val_X_list)

        loss_vals.append(loss_val.item())
        loss_trains.append(loss_train.item())

    #Validate and Test
    best_model.eval()
    output, _, _ = best_model(X_list1 + [features_normalized], X_list2 + [features_normalized], adj_normalized, knn_adj_norm)
    output = torch.log_softmax(output, dim=1)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    all_val.append(acc_val.item())
    all_test.append(acc_test.item())

    print(acc_test.item())
    #
    # UNF_mask = torch.load("./UNF_mask/"+args.dataset+"_UNF_mask.pt")
    # acc_UNF = accuracy(output[UNF_mask], labels[UNF_mask])
    # print("UN_F acc:", acc_UNF.item())
    # all_unf.append(acc_UNF.item())
    #
    # F_mask = torch.load("./UNF_mask/"+args.dataset+"_F_mask.pt")
    # acc_F = accuracy(output[F_mask], labels[F_mask])
    # print("F acc:", acc_F.item())
    # all_f.append(acc_F.item())




print(np.mean(all_test),end=',')


print(np.mean(all_val), np.std(all_val), np.mean(all_test), np.std(all_test))

