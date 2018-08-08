import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        b,c = x.shape
        x = x.view(-1,c)
        y = x.view(14,-1,c)
       # y = y.mean(0)
        y,_ = torch.max(y,0)
        
        #x = (F.log_softmax(x,dim=0))
        return F.log_softmax(y)
