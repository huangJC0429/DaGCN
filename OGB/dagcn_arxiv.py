import argparse
import numpy as np
import random
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from logger import Logger

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = adj@support
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LAGCN(torch.nn.Module):
    def __init__(self, concat, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LAGCN, self).__init__()

        self.convs_initial = torch.nn.ModuleList()
        self.bns_initial = torch.nn.ModuleList()    
        for _ in range(concat):
            self.convs_initial.append(GCNConv(in_channels, hidden_channels, cached=True))
            self.bns_initial.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(concat*hidden_channels, concat*hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(concat*hidden_channels))
        self.convs.append(GCNConv(concat*hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for conv in self.convs_initial:
            conv.reset_parameters()
        for bn in self.bns_initial:
            bn.reset_parameters()

    def forward(self, x_list, adj_t):
        hidden_list = []
        for i, conv in enumerate(self.convs_initial):
            x = conv(x_list[i], adj_t)
            x = self.bns_initial[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class AWS(nn.Module):
    def __init__(self, concat,  nhid):
        super(AWS, self).__init__()
        self.fc_weight1 = nn.Linear(concat * nhid, 1)
        self.fc_weight2 = nn.Linear(concat * nhid, 1)
    def forward(self, x1, x2):
        '''
        output: lambda1, lambda2
        '''
        lambda1 = F.sigmoid(self.fc_weight1(x1))
        lambda2 = F.sigmoid(self.fc_weight2(x2))
        w = torch.cat([lambda1.view(-1, 1), lambda2.view(-1, 1)], dim=-1)
        w = F.normalize(w, p=1, dim=-1)
        return w[-1, 0], w[-1, 1]

class CLAGCN(nn.Module):
    def __init__(self, concat, nfeat, nhid, nclass, n_layers, dropout):
        super(CLAGCN, self).__init__()
        self.convs_initial = torch.nn.ModuleList()
        self.bns_initial = torch.nn.ModuleList()
        for _ in range(concat):
            self.convs_initial.append(GCNConv(nfeat, nhid, cached=True))
            self.bns_initial.append(torch.nn.BatchNorm1d(nhid))
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(n_layers - 2):
            self.convs.append(
                GCNConv(concat*nhid, concat*nhid, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(concat*nhid))
        self.convs.append(GCNConv(concat*nhid, nclass, cached=True))
        self.dropout = dropout

        self.fc1 = nn.Linear(nclass, nclass)

        self.fc1_weight1 = nn.Linear(concat * nhid, 1)
        self.fc1_weight2 = nn.Linear(concat * nhid, 1)

        self.aws = torch.nn.ModuleList()
        for _ in range(n_layers - 2):
            self.aws.append(
                AWS(concat, nhid))

        self.fc_weight1 = nn.Linear(nclass, 1)
        self.fc_weight2 = nn.Linear(nclass, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for conv in self.convs_initial:
            conv.reset_parameters()
        for bn in self.bns_initial:
            bn.reset_parameters()

    def forward(self, x_list1, x_list2, adj1, adj2):
        hidden_list1 = []
        for i, conv in enumerate(self.convs_initial):
            x1 = conv(x_list1[i], adj1)
            x1 = self.bns_initial[i](x1)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            hidden_list1.append(x1)
        x1 = torch.cat((hidden_list1), dim=-1)

        hidden_list2 = []
        for i, conv in enumerate(self.convs_initial):
            x2 = conv(x_list2[i], adj2)
            x2 = self.bns_initial[i](x2)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            hidden_list2.append(x2)
        x2 = torch.cat((hidden_list2), dim=-1)

        w1, w2 = self.auto_weight1(x1, x2)
        x = w1*x1 + w2*x2
        # x = F.dropout(x, self.dropout, training=self.training)

        for i, conv in enumerate(self.convs[:-1]):
            x1 = conv(x, adj1)
            x1 = self.bns[i](x1)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)

            x2 = conv(x, adj2)
            x2 = self.bns[i](x2)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)

            w1, w2 = self.aws[i](x1, x2)
            x = w1 * x1 + w2 * x2
            # print(w1, w2)
            x = F.dropout(x, self.dropout, training=self.training)


        gnn_prop1 = self.convs[-1](x, adj1)
        gnn_prop2 = self.convs[-1](x, adj2)

        w1, w2 = self.auto_weight(gnn_prop1, gnn_prop2)
        out = w1*gnn_prop1 + w2*gnn_prop2
        return out, gnn_prop1, gnn_prop2# gnn_prop2

    def auto_weight(self, x1, x2):
        '''
        output: lambda1, lambda2
        '''
        lambda1 = F.sigmoid(self.fc_weight1(x1))
        lambda2 = F.sigmoid(self.fc_weight2(x2))
        w = torch.cat([lambda1.view(-1, 1), lambda2.view(-1, 1)], dim=-1)
        w = F.normalize(w, p=1, dim=-1)
        return w[-1, 0], w[-1, 1]

    def auto_weight1(self, x1, x2):
        '''
        output: lambda1, lambda2
        '''
        lambda1 = F.sigmoid(self.fc1_weight1(x1))
        lambda2 = F.sigmoid(self.fc1_weight2(x2))
        w = torch.cat([lambda1.view(-1, 1), lambda2.view(-1, 1)], dim=-1)
        w = F.normalize(w, p=1, dim=-1)
        return w[-1, 0], w[-1, 1]


parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')

parser.add_argument("--concat", type=int, default=1)
parser.add_argument("--samples", type=int, default=2)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--runs', type=int, default=10)

parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1.0, help='Lamda')
parser.add_argument('--k', type=int, default=20, help=' K for knn.')

args = parser.parse_args()
print(args)


def consis_loss(logps, temp=args.tem):
    avg_p = torch.exp(logps[0])

    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = torch.mean((avg_p - sharp_p).pow(2).sum(1))
    return loss


def consis_loss2(out, out1, out2, temp=args.tem):
    logps = [out1, out2]
    # ps = [torch.exp(p) for p in logps]
    ps = [p for p in logps]
    avg_p = out
    avg_p = avg_p

    sharp_p = avg_p# (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return loss
def GCN_norm(edge_index, x):
    # print(edge_index)
    # row, col = edge_index.row, edge_index.col
    # num_nodes = x.shape[0]
    #
    # adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    adj_norm = gcn_norm(edge_index.to_symmetric())
    return adj_norm
def row_l1_normalize(X):
    norm = 1e-6 + X.sum(dim=1, keepdim=True)
    return X/norm
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx["train"].to(device)
# print(len(train_idx))
# exit()

# adj_normalized = GCN_norm(data.adj_t, data.x)
cvae_model = torch.load("model/arxiv_.pkl")
knn_adj = torch.load(f'./knn_graph/arxiv_sims_{args.k}_.pt')

def get_augmented_features(concat):
    X_list = []
    for _ in range(concat):
        z = torch.randn([data.x.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, data.x).detach()
        X_list.append(augmented_features)
    return X_list

model = CLAGCN(args.concat, data.num_features, args.hidden_channels,
              dataset.num_classes, args.num_layers,
              args.dropout).to(device)

evaluator = Evaluator(name='ogbn-arxiv')
logger = Logger(args.runs, args)

for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        model.train()
        optimizer.zero_grad()

        output_list = []
        for k in range(args.samples):
            X_list1 = get_augmented_features(args.concat)
            X_list2 = get_augmented_features(args.concat)
            model_out, gnn_emb1, gnn_emb2 = model(X_list1+[data.x], X_list2+[data.x],
                                                  data.adj_t, knn_adj)

            output_list.append(torch.log_softmax(model_out, dim=-1))

        loss_train = 0.
        for k in range(len(output_list)):
            loss_train += F.nll_loss(output_list[k][train_idx], data.y.squeeze(1)[train_idx])
        
        loss_train = loss_train/len(output_list)

        loss_consis = consis_loss(output_list)
        loss_consis2 = consis_loss2(model_out, gnn_emb1, gnn_emb2)
        loss_train = loss_train + loss_consis + args.lam * loss_consis2 #+ 1.0 * loss_consis  # + 0.1*model.contrasive_loss(gnn_emb1, gnn_emb2)#  + 0.5*model.contrasive_loss(sharp_emb1, gnn_emb2)
        
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_X_list = get_augmented_features(args.concat)
            output, _, _ = model( X_list1 + [data.x], X_list2 + [data.x],
                                                  data.adj_t, knn_adj)
            y_pred = output.argmax(dim=-1, keepdim=True)

            train_acc = evaluator.eval({
                'y_true': data.y[split_idx['train']],
                'y_pred': y_pred[split_idx['train']],
            })['acc']
            valid_acc = evaluator.eval({
                'y_true': data.y[split_idx['valid']],
                'y_pred': y_pred[split_idx['valid']],
            })['acc']
            test_acc = evaluator.eval({
                'y_true': data.y[split_idx['test']],
                'y_pred': y_pred[split_idx['test']],
            })['acc']

            logger.add_result(run, (train_acc, valid_acc, test_acc))
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss_train.item():.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')

    logger.print_statistics(run)
logger.print_statistics()
