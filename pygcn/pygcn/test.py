import torch
import numpy as np
from data_loader import HopkinsDataset 


loader = HopkinsDataset(window = 3 , root_dir = '../../car_dataset_03/' )


a,b,c = loader[1]

print(a)
print(a.shape)
print(a.type())
input()
print(c)
print(c.shape)
print(c[0])
input()