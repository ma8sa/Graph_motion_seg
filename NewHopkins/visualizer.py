import numpy as np
import scipy.sparse as sp
import cv2
import os
import sys







def plot_mat(idx,sigma=50,msg='s',root_dir='./'):


   gt = np.genfromtxt(root_dir + str(idx).zfill(6) + '_gt.txt')
   nf  = gt.shape
   
   ad = sp.load_npz(root_dir + str(idx).zfill(6) + '_adj.npz').todense()
   
   adj = ad[:nf[0],:nf[0]]
   #adj = ad
   inds = gt.argsort()
   
   adj = adj[:,inds]
   adj = adj[inds,:]
   mn = adj.min()

   np.fill_diagonal(adj,mn) 
   mx = adj.max()

   adj = (adj - mn) / ( mx-mn)
   print(gt.max())
   gr = gt.max()
   
   adj = 255 * adj 
   
   cv2.imwrite(str(idx).zfill(6) +'_s'+ str(sigma)+ '_g'+ str(int(gr)) + msg+'.jpg',adj)
  



root_dir = sys.argv[1]

num = len([ x  for x in os.listdir(root_dir) if x.endswith('.npz')])

print(num)

for i in range(num):
    plot_mat(i,500,sys.argv[2],root_dir)
