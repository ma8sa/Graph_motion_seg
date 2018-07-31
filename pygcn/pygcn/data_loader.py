import os
import torch
import numpy as np
from torch.utils.data import Dataset , DataLoader

def encode_onehot(labels):
    classes = [1,2,3] 
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def make_adj(x,y,window,sigma=1.0):
   
    n,_ = x.shape
    ind = [[0,0]]
    val = [1.0]
    N = n * window
    for i in range(N):
        for j in range(N):

            frame_i = i / n
            frame_j = j / n

            idx_i = i % n
            idx_j = j % n

	    # condition for temorapl consistency
            if idx_i == idx_j:
               ind.append([i,j])
               val.append(1.0)
            elif frame_i == frame_j:
               # exp(-1* euc_dist(x,y,x,y)/sigma)
           
               #tmp = np.exp( ((-1) * euc_dist( )) / sigma )
               tmp = 0.0 
               ind.append([i,j])
               val.append(tmp)
                
    ind = torch.LongTensor(ind)
    val = torch.FloatTensor(val)
    print(ind.type())
    print(ind)
    input()         
    #adj = torch.zeros([N,N],dtype=torch.float64)
    adj = torch.sparse.FloatTensor(ind.t(),val,torch.Size([N,N]))
    print(adj)
    input()
    return adj
    
    

class HopkinsDataset(Dataset):



      def __init__(self, window, root_dir, transform=None):

          self.window = window
          self.root_dir = root_dir
          self.transform = transform


      def __len__(self):
         
         return len(os.listdir(self.root_dir))


      def __getitem__(self,idx):
          
          tmp = np.genfromtxt(self.root_dir + str(idx).zfill(6) + '_.txt')
          nf , _ = tmp.shape 
          gt = tmp[:,2*self.window] 
          tmp = tmp[:,:2*self.window] 
          x = tmp[:,0::2]
          y = tmp[:,1::2]

          features = torch.FloatTensor(np.ones((nf,100)))
          #labels = encode_onehot(gt)
          gt = gt -1
          labels = torch.LongTensor(gt)
          
          adj = make_adj(x,y,self.window)

          # make adj matrix 
          # experiment 1 features just has 1 in it
          
          return adj,features,labels
