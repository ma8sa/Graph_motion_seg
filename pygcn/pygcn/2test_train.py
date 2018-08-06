from __future__ import division
from __future__ import print_function

import time
import argparse
import shutil
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from test_models import GCN
from test_loader import HopkinsDataset,to_sparse
from torch.utils.data.sampler import SubsetRandomSampler

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

def save_checkpoint(state, is_best, filename='2test_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '2test_model_best.pth.tar')

def train(epoch,loader,model,optimizer):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    count = 0
    len_train = len(loader) 
    t = time.time()
    for i ,( adj,features,labels )  in enumerate(loader):
          # cuda()
          features = features.cuda()
         # adj = to_sparse(adj)
          adj = adj.cuda()
          labels = labels.cuda()
          # model()
          output = model(features, adj)
          # loss()
          #a,b,c = output.shape
          #output = output.view(-1,c)
          labels = labels.view(-1)
          loss_train = F.nll_loss(output, labels)
          
          # optimize()
          # update()
          count += 1
          total_loss += loss_train

          #if i%40 == 0:
           #  print("# {}/{} loss : {} , time: {} , ETA:  {}".format(i,len_train,total_loss/count,time.time()-t, (len_train-i)/40.0 * (time.time()-t) ))
            # t = time.time()
          
          loss_train.backward()
          optimizer.step()
          


    return (total_loss/count)



def validate(epoch,loader,model):


   
    model.eval()
    total_loss= 0.0
    total_acc = 0.0
    count = 0
    val_len = len(loader)    
    
    t = time.time()

        
    for i ,( adj,features,labels )  in enumerate(loader):
          # cuda() 
          features = features.cuda()
          adj = adj.cuda()
          labels = labels.cuda()
          # model()
          output = model(features, adj)
          labels = labels.view(-1)
          # loss()
          val_train = F.nll_loss(output, labels)
          acc_val = accuracy(output, labels)

          count += 1
          total_loss += val_train
          total_acc += acc_val

          #if i%40 == 0:
             #print("# {} val loss : {}, acc val:{}".format(i,val_train,acc_val)) 
             #print("# {}/{} loss : {} , AVG acc : {}, time: {} , ETA:  {}".format(i,val_len,float(total_loss)/count,total_acc/count,time.time()-t, (val_len-i)/40.0 * (time.time()-t) ))
             #t = time.time()


    return total_loss/count,total_acc/count


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()A
loader = HopkinsDataset(window = 3 , root_dir = '../../train_dataset_03/' )
test_loader = HopkinsDataset(window = 3 , root_dir = '../../val_dataset_03/' )

dataset_len = len(loader)
split = int(dataset_len/4)
indices = list(range(dataset_len))

validation_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(validation_idx))

train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

train_loader = torch.utils.data.DataLoader(loader, 
                batch_size=1,num_workers=8, shuffle=False,sampler=train_sampler )

val_loader = torch.utils.data.DataLoader(loader, 
                batch_size=1,num_workers=8, shuffle=False,sampler=validation_sampler)


model = GCN(nfeat=100,
            nhid=args.hidden,
            nclass=3,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


#for adj, features, labels in loader: 
#tt = adj 
#print(features.shape)
#input()
# Model and optimizer
#  print(adj)
 # input()
 # model.train()
 # optimizer.zero_grad()
 # output = model(features, adj)
 # loss_train = F.nll_loss(output, labels)
 # print(loss_train)
 # input()
 # loss_train.backward()
 # optimizer.step()
 # print(" step one done")
 # input()


if args.cuda:
    model.cuda()
    #features = features.cuda()
    #adj = adj.cuda()
    #labels = labels.cuda()
    #idx_train = idx_train.cuda()
    #idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()



# Train model
t_total = time.time()
best_val_loss = 100000

for epoch in range(args.epochs):
    t = time.time()
    print(" epoch # {}".format(epoch))

    train_loss = train(epoch,train_loader,model,optimizer)
    print(" Train Loss = {}".format(train_loss))

    val_loss, acc = validate(epoch,val_loader,model)
    print(" val Loss = {}, accuracy : {}".format(val_loss,acc))
    print(" epoch time {} epoch left {} Time left {}".format(time.time() - t, args.epochs - epoch , ( args.epochs - epoch ) * ( time.time() - t)))
    is_best=False

    if val_loss < best_val_loss:
       best_val_loss = val_loss
       is_best = True
       
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val': val_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
 
       
    
#print("Optimization Finished!")
#print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
