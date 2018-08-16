import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, int(nhid/2))
 
        self.gcy1 = GraphConvolution(nfeat, nhid)
        self.gcy2 = GraphConvolution(nhid, int(nhid/2))

        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, int(nhid/2))
        self.gc5 = GraphConvolution(int(nhid/2), nclass)
        self.dropout = dropout

    def forward(self, x,y, adj, adj2):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        y = F.relu(self.gcy1(y, adj))
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.relu(self.gcy2(y, adj))
       
        x = torch.cat([y,x],1) 
        x = self.gc3(x,adj)
        x = self.gc4(x,adj)
        x = self.gc5(x,adj)
        b,c = x.shape
        x = x.view(14,-1,c)
        x,_ = torch.max(x,0)
        x = (F.log_softmax(x,dim=1))
        return x
