import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, int(nhid/2))
        self.gc3 = GraphConvolution(int(nhid/2), int(nhid/2))
        self.gc4 = GraphConvolution(int(nhid/2), nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.gc3(x,adj)
        b,c = x.shape
        x = x.view(14,-1,c)
        a,b,c = x.shape
        x,_ = torch.max(x,0)
        adj = adj.to_dense()
        adjj = adj.narrow(0,0,b)
        adjj = adjj.narrow(1,0,b)
        x = self.gc4(x,adjj)
            
        x = (F.log_softmax(x,dim=1))
        return x
