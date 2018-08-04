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
from pygcn.models import GCN
from data_loader import HopkinsDataset
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
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def train(epoch,loader,model,optimizer):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    count = 0
    len_train = len(loader) 
    for i   in range(len_train):
          adj,features,labels = loader[i] 
          # cuda()
          features = features.cuda()
          adj = adj.cuda()
          labels = labels.cuda()
          # model()
          output = model(features, adj)
          # loss()
          loss_train = F.nll_loss(output, labels)
          
          # optimize()
          # update()
          if i%40 == 0:
             print("# {}/{} loss : {}".format(i,len_train,loss_train)) 
          
          loss_train.backward()
          optimizer.step()
          
          count += 1
          total_loss += loss_train


    return (total_loss/count)
def validate(epoch,loader,model):
   
    model.eval()
    total_loss= 0.0
    total_acc = 0.0
    count = 0
    val_len = len(loader)    
    

    for i in range(val_len):
        
          # cuda() 
          adj,features,labels = loader[i] 
          features = features.cuda()
          adj = adj.cuda()
          labels = labels.cuda()
          # model()
          output = model(features, adj)
          # loss()
          val_train = F.nll_loss(output, labels)
          acc_val = accuracy(output, labels)

          if i%40 == 0:
             print("# {} val loss : {}, acc val:{}".format(i,val_train,acc_val)) 

          count += 1
          total_loss += val_train
          total_acc += acc_val

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
val_loader = HopkinsDataset(window = 3 , root_dir = '../../val_dataset_03/' )

dataset_len = len(loader)


#train_sampler = SubsetRandomSampler(train_idx)
#validation_sampler = SubsetRandomSampler(validation_idx)

#train_loader = torch.utils.data.DataLoader(loader, 
#                batch_size=1, sampler=train_sampler)

#validation_loader = torch.utils.data.DataLoader(loader, 
 #               batch_size=2, sampler=validation_sampler)


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
    print(" epoch # {}".format(epoch))

    train_loss = train(epoch,loader,model,optimizer)
    print(" Train Loss = {}".format(train_loss))
    val_loss, acc = validate(epoch,val_loader,model)
    print(" val Loss = {}, accuracy : {}".format(val_loss,acc))

    is_best=False

    if val_loss < best_val_loss:
       best_val_loss = val_loss
       is_best = True
       
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': 0,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
 
       
    
#print("Optimization Finished!")
#print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
#test()
