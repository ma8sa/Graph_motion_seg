import os
import torch
import numpy as np
from torch.utils.data import Dataset , DataLoader
import scipy.sparse as sp

def encode_onehot(labels):
    classes = [1,2,3] 
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def euc_dist(x_i,y_i,x_j,y_j):
   
    return np.sqrt( (float(x_i-x_j))**2 + (float(y_i-y_j))**2 )

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def make_adj(x,y,window,sigma=50.0):
   
    n,_ = x.shape
    ind = [[0,0]]
    val = [1.0]
    N = n * (window-1)
    adj = np.zeros((N,N),dtype=np.float64)
    for i in range(N):
        for j in range(N):

            frame_i = int(i / n)
            frame_j = int(j / n)

            idx_i = i % n
            idx_j = j % n

            
	    # condition for temorapl consistency
            if idx_i == idx_j:
               #ind.append([i,j])
               #val.append(1.0)
               adj[i,j] = 1.0
               
            elif frame_i == frame_j:
               # exp(-1* euc_dist(x,y,x,y)/sigma)
               euc1 =euc_dist(x[idx_i,frame_i],y[idx_i,frame_i],x[idx_i,frame_i+1],y[idx_i,frame_i+1] )
               euc2 =euc_dist(x[idx_j,frame_j],y[idx_j,frame_j],x[idx_j,frame_j+1],y[idx_j,frame_j+1] )

               tmp = np.exp( ((-1) * ((euc1 - euc2)**2)) / sigma )
               adj[i,j] = tmp
               #tmp = 0.0 
               #ind.append([i,j])
               #val.append(tmp)
    adj = normalize(adj)                
    adj = torch.FloatTensor(adj)
    adj = to_sparse(adj)
    return adj
    
    

class HopkinsDataset(Dataset):



      def __init__(self, window, root_dir, transform=None):

          self.window = window
          self.root_dir = root_dir
          self.transform = transform


      def __len__(self):
         
         return len(os.listdir(self.root_dir))


      def __getitem__(self,idx):
          
          tmp = np.genfromtxt(self.root_dir + str(idx).zfill(6) + '.txt')
          nf , _ = tmp.shape 
          gt = tmp[:,2*self.window] 
          tmp = tmp[:,:2*self.window] 
          x = tmp[:,0::2]
          y = tmp[:,1::2]

          features = torch.FloatTensor(np.ones((nf,100)))
          for _ in range(self.window-2): 
             features = torch.cat((features,features), 0)
             gt = np.concatenate((gt,gt),axis=0)
          #labels = encode_onehot(gt)
          gt = gt -1
          labels = torch.LongTensor(gt)
          
          adj = make_adj(x,y,self.window)

          # make adj matrix 
          # experiment 1 features just has 1 in it
          
          return adj,features,labels
