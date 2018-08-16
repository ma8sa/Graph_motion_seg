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

def make_adj(x,y,window,fl=0,sigma=100.0,sigma2=400.0):

    n,_ = x.shape
    ind = [[0,0]]
    val = [1.0]
    N = n * (window-1)
    adj = np.zeros((window-1,n,n),dtype=np.float64)
    #adj = torch.zeros([N, N], dtype=torch.float64).cuda()
    for i in range(N):
        for j in range(N):

            frame_i = int(i / n)
            frame_j = int(j / n)

            idx_i = i % n
            idx_j = j % n

            # condition for temorapl consistency
               #ind.append([i,j])
               #val.append(1.0)

            if frame_i == frame_j:
               # exp(-1* euc_dist(x,y,x,y)/sigma)
             #  euc1 =euc_dist(x[idx_i,frame_i],y[idx_i,frame_i],x[idx_i,frame_i+1],y[idx_i,frame_i+1] )
              # euc2 =euc_dist(x[idx_j,frame_j],y[idx_j,frame_j],x[idx_j,frame_j+1],y[idx_j,frame_j+1] )

               Xi = x[idx_i,frame_i] - x[idx_i,frame_j+1]
               Yi = y[idx_i,frame_i] - y[idx_i,frame_j+1]

               Xj = x[idx_j,frame_i] - x[idx_j,frame_i+1]
               Yj = y[idx_j,frame_i] - y[idx_j,frame_i+1]

               d2 = euc_dist(Xi,Yi,Xj,Yj)
               
               d1 = euc_dist(x[idx_i,frame_i],y[idx_i,frame_i],x[idx_j,frame_j],y[idx_j,frame_j])

               #tmp = np.exp( ((-1) * ((euc1 - euc2)**2)) / sigma )
               
               an1 = np.arctan(float(Yi)/float(abs(Xi)+1))
               an2 = np.arctan(float(Yj)/float(abs(Xj)+1))
               if fl == 0:
                  tmp = ( np.exp( ((-1) * ((d2)**2)) / sigma )  + np.exp( ((-1)*(an1-an2)**2)/sigma2 ) + np.exp( ((-1)*(d1)**2)/50.0 ))/3.0
               if fl == 1:
                  tmp = ( np.exp( ((-1) * ((d2)**2)) / sigma )  )
               if fl == 2:
                  tmp = (  np.exp( ((-1)*(an1-an2)**2)/sigma2 ) )
               adj[frame_i,idx_i,idx_j] = tmp
               #tmp = 0.0 
               #ind.append([i,j])
               #val.append(tmp)
    #mn = adj.min()
    #np.fill_diagonal(adj,mn)
    #mx = adj.max()
    #adj = (adj - mn) / ( mx - mn)
    print("adj")
    adj = np.mean(adj,axis=0)
    #adj = [ [ 0 if y < 0.6 else y for y in x] for x in adj] 
    #adj = np.array(adj)
    
    adj = normalize(adj)
#    adj = to_sparse(adj)
    return adj

def mat_to_txt(filename,non_b,count,train=True, window=10,jump=10  ):
    
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
        tmp2 = points[1,:,i:(i+window)]
        adj =  make_adj(tmp1,tmp2,window,1)
        adj2 =  make_adj(tmp1,tmp2,window,2)
        tmp = np.empty ((features,(window-1)*2),dtype=tmp1.dtype) 
        tmp_loc = np.empty ((features,2 ),dtype=tmp1.dtype) 
        tmp_loc[:,0] = tmp1[:,0]
        tmp_loc[:,1] = tmp2[:,0]
        
        tmp_f = gt-1
        st_l = gt.max()
        inds = [ i for i,x in enumerate(tmp_f) if x <= st_l]

        s_inds = np.random.choice(inds, int(len(inds)/2) , replace=False )
        for i in s_inds:
            tmp_f[i] = 0.5

        tmp_f[ tmp_f[:] < st_l ] = 0.5
        
        tmp_loc = np.concatenate((tmp_loc,tmp_f),axis=1) 
        
        tmp1 = np.array([x - tmp1.T[i - 1] for i, x in enumerate(tmp1.T) if i > 0]       )
        tmp2 = np.array([x - tmp2.T[i - 1] for i, x in enumerate(tmp2.T) if i > 0]       )
        
        tmp1 = tmp1.T
        tmp2 = tmp2.T

        tmp[:,0::2] = tmp1
        tmp[:,1::2] = tmp2
        s_adj = sp.csc_matrix(adj)
        s_adj2 = sp.csc_matrix(adj2)
        #tmp = np.concatenate((tmp,gt),axis=1)
        np.savetxt(folder + '/' + str(count).zfill(6)+'_f.txt',(tmp))
        np.savetxt(folder + '/' + str(count).zfill(6)+'_f2.txt',(tmp_loc))
        np.savetxt(folder + '/' + str(count).zfill(6)+'_gt.txt',(gt))
        #np.savetxt(folder + '/' + str(count).zfill(6)+'_adj.txt',(adj))
        sp.save_npz(folder + '/' + str(count).zfill(6)+'_adj.npz',(s_adj))
        sp.save_npz(folder + '/' + str(count).zfill(6)+'_adj_an.npz',(s_adj2))
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
   
   for f in val:
      print( " -------------------------- file name {}-------------------".format(f))
      count = mat_to_txt(f,non_b,count,False)
   count = 0 
   for f in train:
      print( " -------------------------- file name {}-------------------".format(f))
      count = mat_to_txt(f,non_b,count)
   with open('./non_b.txt', 'w') as file_handler:
    for item in non_b:
        file_handler.write("{}\n".format(item))
   
  
