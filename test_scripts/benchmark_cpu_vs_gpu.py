"""
This script is to motivate the use of CPU vs GPU for RL problems.
Two networks are typically encountered in RL problems -- one for the actor (policy)
and one for the critic (baseline). The actor often impliments some form of weighted
maximum-likelihood optimization (from a computational sense), with the weights specified
by the critic. The problem for the critic is a pure supervised learning problem.

Thus we can study the performance difference by studying supervised learning for small networks.
"""

from os import environ
# Select GPU 0 only
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['CUDA_VISIBLE_DEVICES']='0'
environ['MKL_THREADING_LAYER']='GNU'

import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import time as timer
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Load data
xdata = pickle.load(open('./data_cpu_gpu/inputs.pickle', 'rb'))
ydata = pickle.load(open('./data_cpu_gpu/targets.pickle', 'rb'))
xdata = xdata.astype('float32')
ydata = ydata.astype('float32')

def make_model(num_inputs, use_gpu=False):
    model = nn.Sequential()
    model.add_module('fc_0', nn.Linear(num_inputs, 128))
    model.add_module('relu_0', nn.ReLU())
    model.add_module('fc_1', nn.Linear(128, 128))
    model.add_module('relu_1', nn.ReLU())
    model.add_module('fc_2', nn.Linear(128, 1))
    if use_gpu:
        model.cuda()
    return model

def train_step(model, optimizer, loss_function, xdata, ydata, use_gpu=False):
    model.train()
    if use_gpu:
        xvar = Variable(torch.from_numpy(xdata).float().cuda(), requires_grad=False)
        yvar = Variable(torch.from_numpy(ydata).cuda(), requires_grad=False)
    else:
        xvar = Variable(torch.from_numpy(xdata).float(), requires_grad=False)
        yvar = Variable(torch.from_numpy(ydata), requires_grad=False)
    optimizer.zero_grad()
    yhat = model(xvar)
    loss = loss_function(yhat, yvar)
    if use_gpu:
        batch_loss = loss.cpu().data.numpy().ravel()[0]
    else:
        batch_loss = loss.data.numpy().ravel()[0]
    loss.backward()
    optimizer.step()
    return batch_loss

def measure_error(model, loss_function, xdata, ydata, use_gpu=False):
    model.eval()
    if use_gpu:
        xvar = Variable(torch.from_numpy(xdata).float().cuda(), requires_grad=False)
        yvar = Variable(torch.from_numpy(ydata).cuda(), requires_grad=False)
    else:
        xvar = Variable(torch.from_numpy(xdata).float(), requires_grad=False)
        yvar = Variable(torch.from_numpy(ydata), requires_grad=False)
    yhat = model(xvar)
    loss = loss_function(yhat, yvar)
    if use_gpu:
        loss_val = loss.cpu().data.numpy().ravel()[0]
    else:
        loss_val = loss.data.numpy().ravel()[0]
    # compute relative error
    rel_error = loss_val/np.mean(ydata**2)
    return loss_val, rel_error

# Training loop
# -------------------
USE_GPU = False
epochs = 10
batch_size = 64

model = make_model(xdata.shape[1], USE_GPU)
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.MSELoss()

ts = timer.time()
loss_before, relative_error = measure_error(model, loss_function, xdata, ydata, USE_GPU)
print("Epoch = %i | Loss = %f | Relative error = %f | time = %f" %
      (0, loss_before, relative_error, timer.time() - ts))

for ep in range(epochs):
    rand_idx = np.random.permutation(xdata.shape[0])
    ts = timer.time()
    for mb in range(int(xdata.shape[0]/batch_size)-2):
        batch_x = xdata[rand_idx[mb*batch_size: (mb+1)*batch_size]]
        batch_y = ydata[rand_idx[mb*batch_size: (mb+1)*batch_size]]
        batch_loss = train_step(model, optimizer, loss_function, batch_x, batch_y, USE_GPU)

    epoch_loss, relative_error = measure_error(model, loss_function, xdata, ydata, USE_GPU)
    print("Epoch = %i | Loss = %f | Relative error = %f | time = %f" %
          (ep+1, epoch_loss, relative_error, timer.time()-ts))