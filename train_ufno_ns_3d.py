"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such that the Navier-Stokes equation.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

import time
from timeit import default_timer

from fourier_3d_unet import U_FNO_3d

import os
import h5py

################################################################
# configs
################################################################
TRAIN_PATH = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'
TEST_PATH = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'

ntrain = 1000
ntest = 200

modes = 8
width = 20

batch_size = 10
learning_rate = 0.001
epochs = 50
iterations = epochs*(ntrain//batch_size)

path = 'ufno_ns_3d'
path_model = '/content/drive/MyDrive/fno_models/'
model_path = path_model + path
#... existing code ...
#... existing code ...
#... existing company ...
#... existing code ...
#... existing code ...
# For this experiment, the time steps are organized as (1000, 64, 64, 30),
# where the last dimension is time. We will use the first 10 time steps
# to predict the next 10 time steps.
sub = 1
S = 64 // sub
T_in = 10
T = 10
step = 1


################################################################
# load data
################################################################

#... existing code ...
#... existing code ...
#... existing company ...
#... existing code ...
#... existing code ...
try:
    with h5py.File(TRAIN_PATH, 'r') as f:
        a = f['u'][:]
except KeyError:
    print(f"Error: 'u' key not found in {TRAIN_PATH}. Please check the MAT file structure.")
    exit()
except Exception as e:
    print(f"An error occurred while reading {TRAIN_PATH}: {e}")
    exit()

train_a = torch.tensor(a[:ntrain, ::sub, ::sub, :T_in], dtype=torch.float)
train_u = torch.tensor(a[:ntrain, ::sub, ::sub, T_in:T_in+T], dtype=torch.float)

#... existing code ...
#... existing code ...
#... existing company ...
#... existing code ...
#... existing code ...
test_a = torch.tensor(a[-ntest:, ::sub, ::sub, :T_in], dtype=torch.float)
test_u = torch.tensor(a[-ntest:, ::sub, ::sub, T_in:T_in+T], dtype=torch.float)


print(train_a.shape)
print(train_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])


train_a = train_a.reshape(ntrain, S, S, 1, T_in).repeat([1, 1, 1, S, 1])
test_a = test_a.reshape(ntest, S, S, 1, T_in).repeat([1, 1, 1, S, 1])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

################################################################
# model and optimizer
################################################################
model = U_FNO_3d(modes, modes, modes, width, in_channels=T_in, out_channels=T).cuda()
# model = torch.load('model/ns_fourier_3d_N1000_ep100_m8_w20')

print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)


myloss = LpLoss(size_average=False)
for ep in range(epochs):
#... existing code ...
#... existing code ...
#... existing company ...
#... existing code ...
#... existing code ...
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        # The U-FNO model expects a 5D tensor (batch, H, W, D, T_in)
        # x is already in the correct shape: (batch, S, S, S, T_in)

        optimizer.zero_grad()
        out = model(x)
        
        # The output of the model is (batch, S, S, S, T)
        # We need to compare it with y, which is (batch, S, S, T).
        # We must expand y to match the output shape.
        y = y.reshape(batch_size, S, S, 1, T).repeat([1, 1, 1, S, 1])

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += loss.item()
        
#... existing code ...
#... existing code ...
#... existing company ...
#... existing code ...
#... existing code ...
    test_l2 = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            # Input x is already (batch, S, S, S, T_in)
            out = model(x)

            # Expand y to match the output shape for loss calculation.
            y = y.reshape(batch_size, S, S, 1, T).repeat([1, 1, 1, S, 1])

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2/= ntrain
    test_l2 /= ntest
#... existing code ...
#... existing code ... 