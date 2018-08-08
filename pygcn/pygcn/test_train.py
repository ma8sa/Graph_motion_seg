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
from tensorboardX import SummaryWriter
import gc

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.015,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
def memReport():
    cc = 0
    for obj in gc.get_objects():
        
        if torch.is_tensor(obj) :
          #  print(obj.name,type(obj), obj.size())
             cc += 1

    print(" num 0f obj : ",cc)
#    input()


def save_checkpoint(state, is_best, filename='test_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'test_model_best.pth.tar')

def train(epoch,loader,model,optimizeri,writer):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    len_train = len(loader) 
    t = time.time()
    #for i ,( adj,features,labels )  in enumerate(loader):
    for i  in range(len(loader)):
          # cuda()
          adj,features,labels = loader[i]
          #adj.unsqueeze_(0)
          #features.unsqueeze_(0)
         
          optimizer.zero_grad()

          features = features.cuda()
         # adj = to_sparse(adj)
          adj = adj.cuda()
          labels = labels.cuda()
          # model()
          output = model(features, adj)
          # loss()
          #a,b,c = output.shape
          #output = output.view(-1,c)
          loss_train = F.nll_loss(output, labels)
          # optimize()
          # update()
          total_loss += loss_train.data 
          count += 1
          #total_loss += loss_train

          if i%40 == 0:
             print("# {}/{} loss : {} , time: {} , ETA:  {}".format(i,len_train,count,time.time()-t, (len_train-i)/40.0 * (time.time()-t) ))
             writer.add_scalar('train_loss', (total_loss.data)/count,(epoch*len_train + i)/40 )
             t = time.time()
          
          loss_train.backward()
          optimizer.step()
          #memReport() 
          #gc.collect()
          del adj,features,labels

  #  return (total_loss.data/count)

def test(model,loader,resume):
   
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
     
    for i  in range(len(loader)):
          # cuda()
          adj,features,labels = loader[i]
          adj.unsqueeze_(0)
          features.unsqueeze_(0)
          # cuda() 
          features = features.cuda()
          adj = adj.cuda()
          labels = labels.cuda()
          
          output = model(features, adj)
          print(" output") 
          print(output)
          print(" checking arg max")
          print( output[ output[:,0] > output[:,1] ] )
          a = torch.argmax( output,dim=1)
          print(" zero ")
          print( a [ a[:] == 0 ] )
          input()
          print("argmax")
          print(a) 
          b = torch.eq(a,labels)
          print(labels.shape)
          print(a.shape)
          print("eq")
          print(b)
          print("accuracy") 
          print(torch.sum(b))
          
          print(float(torch.sum(b)/len(labels)))
          input()
          
          



def validate(epoch,loader,model,writer):


   
    model.eval()
    total_loss= 0.0
    total_acc = 0.0
    count = 0
    
    t = time.time()

    val_len = len(loader) 
    #for i ,( adj,features,labels )  in enumerate(loader):
    for i  in range(len(loader)):
          # cuda()
          adj,features,labels = loader[i]
         #adj.unsqueeze_(0)
          #features.unsqueeze_(0)
          # cuda() 
          features = features.cuda()
          adj = adj.cuda()
          labels = labels.cuda()
          # model()
          output = model(features, adj)
          # loss()
          val_train = F.nll_loss(output, labels)
          a = torch.argmax( output,dim=1)
          b = torch.eq(a,labels)
          c = (torch.sum(b))
          d = labels.shape
          acc = (c.data * 100)/d[0]
          total_acc += acc.data
          total_loss += val_train.data 
          count += 1

          if i%10 == 0:
            print("# {}/{} loss : {} , AVG acc : {}, time: {} , ETA:  {}".format(i,val_len,count,acc,time.time()-t, (val_len-i)/40.0 * (time.time()-t) ))
             #t = time.time()

          gc.collect()
          del adj,labels,features,val_train
    #writer.add_scalar('val_acc', acc.data, epoch)
    return total_loss.data/count,total_acc/count
    


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = True
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("loading  ")
# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()A
loader = HopkinsDataset(window = 15 , root_dir = '../../train_dataset_15/' )
print("len of dataset {}".format(len(loader)))
test_loader = HopkinsDataset(window = 15 , root_dir = '../../val_dataset_15/' )

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
            nclass=2,
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
 # optimizer.step()
 # print(" step one done")
 # input()

writer = SummaryWriter()
model.cuda()
#if args.cuda:
  #  model.cuda()
    #features = features.cuda()
    #adj = adj.cuda()
    #labels = labels.cuda()
    #idx_train = idx_train.cuda()
    #idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()


#test(model,test_loader,'test_checkpoint.pth.tar')

# Train model
t_total = time.time()
best_val_loss = 100000

for epoch in range(args.epochs):
    t = time.time()
    print(" epoch # {}".format(epoch))

    #train_loss  = train(epoch,loader,model,optimizer)
    print("pre training mem ")
    print(torch.cuda.max_memory_allocated()/8000000) 
    #input()
    train(epoch,loader,model,optimizer,writer)
    print("post training mem ")
    print(torch.cuda.max_memory_allocated()/1000000) 
#    print(" Train Loss = {} ".format(train_loss))
    val_loss = 0.0
    val_loss,acc = validate(epoch,test_loader,model,writer)
    writer.add_scalar('val_loss', val_loss.data,epoch)
    writer.add_scalar('val_acc', acc.data,epoch )
    #val_loss, acc = validate(epoch,val_loader,model)
    print(" val Loss = {}, accuracy : {}".format(val_loss,acc))
  #  print(" epoch time {} epoch left {} Time left {}".format(time.time() - t, args.epochs - epoch , ( args.epochs - epoch ) * ( time.time() - t)))
  #  print(type(train_loss),type(val_loss),type(acc))
  #  input()
    is_best=False
    #writer.add_scalar('train_loss', train_loss.data,epoch )
    #writer.add_scalar('val_loss', val_loss.data,epoch )
    #writer.add_scalar('val_acc', acc.data, epoch)
    if val_loss < best_val_loss:
       best_val_loss = val_loss
       is_best = True
       
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
           # 'best_val': val_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    memReport()
    gc.collect() 
       
#writer.export_scalars_to_json("./all_scalars.json")
#writer.close() 
#print("Optimization Finished!")
#print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
