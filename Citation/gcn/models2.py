import torch.nn as nn
import torch.nn.functional as F
import torch
from gcn.layers import GraphConvolution, MLPLayer
from torch_geometric.utils import one_hot, spmm


class GCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, tau=0.0):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, _, adj, _1):
        x = F.dropout(x[0], self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x, None, None

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid)
        self.layer2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x)
        return  torch.nn.functional.log_softmax(x, dim=1)
class LP(torch.nn.Module):
    def __init__(self, y, train_mask, K, alpha=0.9, temp=1):
        super(LP, self).__init__()
        self.y = y
        self.train_mask = train_mask
        self.alpha = alpha
        self.temp = temp
        self.num_layers = K
    def forward(self,  homo_adj):
        # return self.LLP(self.y, homo_adj, mask=self.train_mask)

        # out = self.LP(self.y, homo_adj, mask=self.train_mask)
        if self.y.dtype == torch.long and self.y.size(0) == self.y.numel():
            self.y = one_hot(self.y.view(-1))
        # out = self.y

        out = torch.zeros_like(self.y).float()
        out[self.train_mask] = self.y[self.train_mask]
        # print(out.shape)
        # print(homo_adj)

        for _ in range(self.num_layers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            out = self.alpha*(homo_adj@out) + (1-self.alpha)*out
            # fix label y on each epoch
            out[self.train_mask] = self.y[self.train_mask]
            out = torch.clip(out, 0, 1)

        return out

class DaGCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, tau=0.9):
        super(DaGCN, self).__init__()
        self.gcn1_list = nn.ModuleList()
        concat = 2
        for _ in range(concat):
            self.gcn1_list.append(GraphConvolution(nfeat, nhid))
        self.gc2 = GraphConvolution(concat * nhid, nclass)
        self.dropout = dropout

        self.fc1 = nn.Linear(nclass, nclass)

        self.fc1_weight1 = nn.Linear(concat * nhid, 1)
        self.fc1_weight2 = nn.Linear(concat * nhid, 1)

        self.fc_weight1 = nn.Linear(nclass, 1)
        self.fc_weight2 = nn.Linear(nclass, 1)
        self.args = args
        self.tau = tau

    def forward(self, x_list1, x_list2, adj1, adj2):
        hidden_list = []
        for k, con in enumerate(self.gcn1_list):
            x = F.dropout(x_list1[k], self.dropout, training=self.training)
            hidden_list.append(F.relu(con(x, adj1)))
        x1 = torch.cat((hidden_list), dim=-1)

        hidden_list2 = []
        for k, con in enumerate(self.gcn1_list):
            x = F.dropout(x_list2[k], self.dropout, training=self.training)
            hidden_list2.append(F.relu(con(x, adj2)))
        x2 = torch.cat((hidden_list2), dim=-1)

        # w1, w2 = self.auto_weight1(x1, x2)
        # x = w1*x1 + w2*x2
        w = self.auto_weight1(x1, x2)
        # print(w[:, 0].shape)
        # print(x1.shape, x2.shape)
        x = (w[:, 0].unsqueeze(1)* x1) + (w[:, 1].unsqueeze(1) * x2)
        x1 = F.dropout(x, self.dropout, training=self.training)
        gnn_prop1 = self.gc2(x1, adj1)

        x2 = F.dropout(x, self.dropout, training=self.training)
        gnn_prop2 = self.gc2(x2, adj2)

        # w1_2, w2_2 = self.auto_weight(gnn_prop1, gnn_prop2)
        # out = w1_2*gnn_prop1 + w2_2*gnn_prop2
        w2 = self.auto_weight(gnn_prop1, gnn_prop2)
        out = (w2[:, 0].unsqueeze(1) * gnn_prop1) + (w2[:, 1].unsqueeze(1) * gnn_prop2)
        return out, gnn_prop1, gnn_prop2

    def get_node_representation(self, x_list1, x_list2, adj1, adj2):
        hidden_list = []
        for k, con in enumerate(self.gcn1_list):
            x = F.dropout(x_list1[k], self.dropout, training=self.training)
            hidden_list.append(F.relu(con(x, adj1)))
        x1 = torch.cat((hidden_list), dim=-1)

        hidden_list2 = []
        for k, con in enumerate(self.gcn1_list):
            x = F.dropout(x_list2[k], self.dropout, training=self.training)
            hidden_list2.append(F.relu(con(x, adj2)))
        x2 = torch.cat((hidden_list2), dim=-1)

        # w1, w2 = self.auto_weight1(x1, x2)
        # x = w1*x1 + w2*x2
        w = self.auto_weight1(x1, x2)
        # print(w[:, 0].shape)
        # print(x1.shape, x2.shape)
        x = (w[:, 0].unsqueeze(1) * x1) + (w[:, 1].unsqueeze(1) * x2)
        return x
    def auto_weight(self, x1, x2):
        '''
        output: lambda1, lambda2
        '''
        lambda1 = F.sigmoid(self.fc_weight1(x1))
        lambda2 = F.sigmoid(self.fc_weight2(x2))
        w = torch.cat([lambda1.view(-1, 1), lambda2.view(-1, 1)], dim=-1)
        if self.args.dataset != "citeseer":
            w = F.normalize(w, p=1, dim=-1)  # citeseer不归一化好一些
        # print(w.shape)
        # exit()
        return w# [-1, 0], w[-1, 1]

    def auto_weight1(self, x1, x2):
        '''
        output: lambda1, lambda2
        '''
        lambda1 = F.sigmoid(self.fc1_weight1(x1))
        lambda2 = F.sigmoid(self.fc1_weight2(x2))
        w = torch.cat([lambda1.view(-1, 1), lambda2.view(-1, 1)], dim=-1)

        w = F.normalize(w, p=1, dim=-1)
        return w# [-1, 0], w[-1, 1]