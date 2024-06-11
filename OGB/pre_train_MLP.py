import numpy as np

import torch
import torch.nn.functional as F

from Citation.utils import accuracy, get_dataset, get_dataset_split
from Citation.gcn.models import MLP, LP
from Citation.early_stop import EarlyStopping, Stop_args
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='ogbn-arxiv',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--num_epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--patience', type=int, default=400,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
# torch.manual_seed(0)
def train_MLP(args, data, train_mask, val_mask, test_mask):
    # Model and optimizer
    model = MLP(args.num_features, args.hidden_dim, args.num_classes, args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)

    def train(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(data.x)
        # print(output.shape)
        # print(data.y.shape)
        loss_train = F.nll_loss(output[train_mask],
                                data.y[train_mask])  # + args.weight_decay * torch.sum(model.linear1.weight ** 2) / 2
        loss_train.backward()
        optimizer.step()

        acc_train = accuracy(output[train_mask], data.y[train_mask])

        # Evaluate validation set performance separately,
        model.eval()
        output = model(data.x)

        loss_val = F.nll_loss(output[val_mask], data.y[val_mask])
        acc_val = accuracy(output[val_mask], data.y[val_mask])

        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #       'acc_train: {:.4f}'.format(acc_train.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'acc_val: {:.4f}'.format(acc_val.item()))

        return loss_val.item(), acc_val.item()

    def A_test():
        model.eval()
        output = model(data.x)
        loss_test = F.nll_loss(output[test_mask], data.y[test_mask])
        acc_test = accuracy(output[test_mask], data.y[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item(), output.exp()

    stopping_args = Stop_args(patience=args.patience, max_epochs=args.num_epochs)
    early_stopping = EarlyStopping(model, **stopping_args)
    for epoch in range(args.num_epochs):
        loss_val, acc_val = train(epoch)
        if early_stopping.check([acc_val, loss_val], epoch):
            break

    print("Optimization Finished!")

    # Restore best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    model.load_state_dict(early_stopping.best_state)
    acc_test, logits_MLP = A_test()
    print(acc_test)
    return model, logits_MLP


def pick_unfriendly_nodes(MLP_model, train_mask, data):
    '''
    Two_point to define the Incompatible GCN Nodes(InG nodes):
    1. Feature propagate and Label propagate are inconsistent.
    2. Label propagation can not reach.

    output:
    InG_mask: InG nodes mask
    preserve_logits: N*c, other nodes logits for SGC or LPA. The row belong to InG nodes are [0,0,...,0]
    '''
    MLP_model.eval()
    # print(MLP_model.training)
    x, y, edge_index = data.x, data.y.long(), data.edge_index
    row, col = edge_index
    num_nodes = x.shape[0]
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    adj_norm = gcn_norm(adj.to_symmetric(), add_self_loops=True)
    # adj_norm = MLP_model.adj_norm
    logits_MLP = MLP_model(x).exp()  # since utilize the log softmax

    if args.dataset in ["citeseer", "pubmed"]:
        lp = LP(y, train_mask, K=5, alpha=0.9)  # for citeseer,K=5, cora=3
    else:
        lp = LP(y, train_mask, K=3, alpha=0.9)  # for citeseer,K=5, cora=3
    logits_lp = lp(adj_norm)

    # For point 1
    MASK1 = logits_lp.max(1)[1] == (adj_norm@adj_norm@logits_MLP).max(1)[1].to(device) # logits_MLP.max(1)[1].to(device)

    return MASK1, adj_norm@adj_norm@logits_MLP

def confidence_gap(logits):
    temp_max = torch.max(logits, 1)[0].reshape(-1, 1)
    b = (temp_max - logits)
    # print(b)
    # exit()
    c_gap = torch.min(torch.where(b == 0, 1 , b), 1)[0]
    return c_gap


def index2bool(index, len):
    if max(index) > len:
        print("error! index out of mask!")
        exit()
    mask = torch.zeros(len).to(device)
    mask[index] = 1
    return mask.bool()

def label_reuse(logit_SGC, y, train_mask, friendly_mask):
    '''
    1. SGC max(logit) is top 30% and conf-conf_gap <0.07
    2. these node should in freind nodes.
    for un friend node label:
    '''
    conf = torch.max(logit_SGC, dim=1)[0]
    conf_gap = confidence_gap(logit_SGC)
    pseudo_y = torch.max(logit_SGC, dim=1)[1]
    pseudo_y[train_mask] = y[train_mask]

    conf[friendly_mask & (~train_mask)] = -999
    index = torch.topk(conf, int(0.3*len(conf)))[1] # 返回索引
    mask = index2bool(index, len(conf)) & ((conf-conf_gap) < 0.07)
    mask = mask | train_mask
    return pseudo_y, mask

path = "data"
# Load data
# adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
dataset, evaluator = get_dataset(
    name=args.dataset,
    root_dir=path,
)
data = dataset._data.to(device)

epoch = 0
if args.dataset in ["cora", "citeseer", "pubmed", "ogbn-arxiv"]:
    adj, features, idx_train, idx_val, idx_test, labels = data.edge_index, data.x, data.train_mask, data.val_mask, data.test_mask, data.y.long().squeeze()
else:
    adj, features,labels = data.edge_index, data.x,  data.y.long().squeeze()
    idx_train, idx_val, idx_test = get_dataset_split(args.dataset, data, None, epoch)
data.y = data.y.long().squeeze()
args.num_features = features.shape[1]
args.num_classes = max(labels)+1
MLP, logits_MLP = train_MLP(args, data, idx_train, idx_val, idx_test)
F_mask, logits_SGC = pick_unfriendly_nodes(MLP, idx_train, data)
idx_test = idx_test.to(device)
UNF_mask = (~F_mask)&(idx_test)
F_mask = (F_mask)&(idx_test)
acc_SGC = (data.y == logits_SGC.max(1)[1])[F_mask].int().sum()/(F_mask.int().sum())
print(acc_SGC)
# exit()
# pseudo_y, new_train_mask = label_reuse(logits_SGC, labels, idx_train, UNF_mask)
# print("F_acc:", accuracy(logits_SGC[UNF_mask&(~idx_train)], data.y[UNF_mask&(~idx_train)]))
# print(pseudo_y, new_train_mask)
# pesudo_acc = ((pseudo_y == data.y)[new_train_mask].int().sum() - 140) / (new_train_mask.int().sum()-140)
# print(pesudo_acc)
# print(new_train_mask.int().sum())

# torch.save(new_train_mask, "./feature/new_train_mask.pt")
# # torch.save(pseudo_y, "./feature/pseudo_y.pt")
if args.dataset in ["cora", "citeseer", "pubmed"]:
    torch.save(UNF_mask, "./UNF_mask/"+args.dataset+"_UNF_mask.pt")
    torch.save(F_mask, "./UNF_mask/"+args.dataset+"_F_mask.pt")
else:
    torch.save(UNF_mask, "./UNF_mask/" + args.dataset + "_UNF_mask"+str(epoch)+".pt")
    torch.save(F_mask, "./UNF_mask/" + args.dataset + "_F_mask"+str(epoch)+".pt")