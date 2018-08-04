import numpy as np
import scipy.io as sio
import os



def mat_to_txt(filename,count,train=True window=3  ):
    
    mat = sio.loadmat(filename)
    
    points = mat['y']
    points = points[:2,:,:]
    gt = mat['s']

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
    
    for i in range(frames-window):
        tmp1 = points[0,:,i:i+window]
        tmp2 = points[1,:,i:i+window]
        tmp = np.empty ((features,window*2),dtype=tmp1.dtype) 
        tmp[:,0::2] = tmp1
        tmp[:,1::2] = tmp2
        tmp = np.concatenate((tmp,gt),axis=1)
        np.savetxt(folder + '/' + str(count).zfill(6)+'.txt',(tmp))
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
   split = int(len_data/4)
   val = np.random.choice( fl, size=split, replace=False)
   train = list(set(fl) - set(val)) 
   
   for f in train:
      print( " -------------------------- file name {}-------------------".format(f))
      count = mat_to_txt(f,count)
   count = 0 
   for f in val:
      print( " -------------------------- file name {}-------------------".format(f))
      count = mat_to_txt(f,count,False)
   
  
