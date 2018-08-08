import numpy as np
import scipy.io as sio
import os
import scipy.sparse as sp

def euc_dist(x_i,y_i,x_j,y_j):

    return np.sqrt( (float(x_i-x_j))**2 + (float(y_i-y_j))**2 )


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
    #adj = torch.zeros([N, N], dtype=torch.float64).cuda()
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
               adj[i,j] = 2.0

            elif frame_i == frame_j:
               # exp(-1* euc_dist(x,y,x,y)/sigma)
             #  euc1 =euc_dist(x[idx_i,frame_i],y[idx_i,frame_i],x[idx_i,frame_i+1],y[idx_i,frame_i+1] )
              # euc2 =euc_dist(x[idx_j,frame_j],y[idx_j,frame_j],x[idx_j,frame_j+1],y[idx_j,frame_j+1] )

               Xi = x[idx_i,frame_i] - x[idx_i,frame_j+1]
               Yi = y[idx_i,frame_i] - y[idx_i,frame_j+1]

               Xj = x[idx_j,frame_i] - x[idx_j,frame_i+1]
               Yj = y[idx_j,frame_i] - y[idx_j,frame_i+1]

               d2 = euc_dist(Xi,Yi,Xj,Yj)

               #tmp = np.exp( ((-1) * ((euc1 - euc2)**2)) / sigma )
               tmp = np.exp( ((-1) * ((d2)**2)) / sigma )
               adj[i,j] = tmp
               #tmp = 0.0 
               #ind.append([i,j])
               #val.append(tmp)
    adj = normalize(adj)
#    adj = to_sparse(adj)
    return adj

def mat_to_txt(filename,non_b,count,train=True, window=15,jump=10  ):
    
    mat = sio.loadmat(filename)
    
    points = mat['y']
    points = points[:2,:,:]
    gt = mat['s'] 
    #if gt.min() == (0):
     #  input("found it")
    
    if len(np.unique(gt)) > 3:
       non_b.append(filename) 
       return count
    #if gt.max() == 3:
     #  input(filename)

    print(points.shape) 
    _,features,frames = points.shape
    print(gt.shape) 
    tmp1 = []
    tmp2 = []
    if train:
       folder = 'train_dataset_' + str(window).zfill(2)
    else:
       folder = 'val_dataset_' + str(window).zfill(2)
    if not(os.path.isdir(folder)):
       os.makedirs(folder)
    
    for i in range(int((frames-window)/jump)):
        i = i * jump
        tmp1 = points[0,:,i:i+window]
        tmp2 = points[1,:,i:i+window]
        adj =  make_adj(tmp1,tmp2,window)
        tmp = np.empty ((features,window*2),dtype=tmp1.dtype) 
        tmp[:,0::2] = tmp1
        tmp[:,1::2] = tmp2
        s_adj = sp.csc_matrix(adj)
        #tmp = np.concatenate((tmp,gt),axis=1)
        np.savetxt(folder + '/' + str(count).zfill(6)+'_f.txt',(tmp))
        np.savetxt(folder + '/' + str(count).zfill(6)+'_gt.txt',(gt))
        #np.savetxt(folder + '/' + str(count).zfill(6)+'_adj.txt',(adj))
        sp.save_npz(folder + '/' + str(count).zfill(6)+'_adj.npz',(s_adj))
        count += 1
        print("count : {}".format(count))


    return count
    

def get_files(ext='.mat',dir_path='.'):
    
    files = []
    for f in os.listdir(dir_path):
         if f.endswith(ext):
            files.append(f)
 
    return files
    



if __name__ == "__main__":
  

   count = 0;
   
   fl = get_files() 
   len_data = len(fl)
   split = int(len_data/8)
   val = np.random.choice( fl, size=split, replace=False)
   train = list(set(fl) - set(val)) 
   non_b = []
   
   for f in train:
      print( " -------------------------- file name {}-------------------".format(f))
      count = mat_to_txt(f,non_b,count)
   count = 0 
   for f in val:
      print( " -------------------------- file name {}-------------------".format(f))
      count = mat_to_txt(f,non_b,count,False)
    
   with open('./non_b.txt', 'w') as file_handler:
    for item in non_b:
        file_handler.write("{}\n".format(item))
   
  
