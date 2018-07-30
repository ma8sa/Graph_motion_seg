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

          # make adj matrix 
          # see in which format the data is returend
          # experiment 1 features just has 1 in it
          
          return x,features,labels
