'''Trains a simple convnet on the MNIST dataset.
Gets to 99.2% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
4 seconds per epoch on a GTX 1080 Ti.
'''
from os import environ
# Select GPU 2 only
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['CUDA_VISIBLE_DEVICES']='0'
environ['MKL_THREADING_LAYER']='GNU'

import argparse
import torch
import keras
import torch.nn as nn
import numpy as np
import time as timer
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

USE_GPU = True
batch_size = 128
num_classes = 10
epochs = 12

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

# load data from keras since it has better interface
from keras.datasets import mnist
# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int')
y_test = y_test.astype('int')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Helper function
# ---------------------------
def make_model(num_out):
    model = nn.Sequential()
    model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=64,
                                        kernel_size=3))
    model.add_module('relu1', nn.ReLU())
    model.add_module('conv2', nn.Conv2d(in_channels=64, out_channels=64,
                                        kernel_size=3))
    model.add_module('relu2', nn.ReLU())
    model.add_module('maxpool1', nn.MaxPool2d(kernel_size=2))
    model.add_module('dropout1', nn.Dropout(p=0.5))
    model.add_module('flatten', Flatten())
    model.add_module('fc1', nn.Linear(9216, 128))
    model.add_module('relu3', nn.ReLU())
    model.add_module('dropout2', nn.Dropout(p=0.5))
    model.add_module('fc2', nn.Linear(128, num_out))
    if USE_GPU:
        model.cuda()
    return model

def train_step(model, optimizer, xdata, ydata):
    model.train()
    if USE_GPU:
        xvar = Variable(torch.from_numpy(xdata).float().cuda(), requires_grad=False)
        yvar = Variable(torch.LongTensor(ydata).cuda(), requires_grad=False)
    else:
        xvar = Variable(torch.from_numpy(xdata).float(), requires_grad=False)
        yvar = Variable(torch.LongTensor(ydata), requires_grad=False)
    optimizer.zero_grad()
    yhat = model(xvar)
    loss = loss_function(yhat, yvar)
    if USE_GPU:
        batch_loss = loss.cpu().data.numpy().ravel()[0]
    else:
        batch_loss = loss.data.numpy().ravel()[0]
    batch_accuracy = measure_accuracy(yhat, ydata)
    loss.backward()
    optimizer.step()
    return batch_loss, batch_accuracy

def measure_accuracy(yhat, y):
    model.eval()
    # yhat is torch variable, y is numpy array
    if USE_GPU:
        yhat_data = yhat.cpu().data.numpy()
    else:
        yhat_data = yhat.data.numpy()
    # find class with largest score
    pred_class = np.argmax(yhat_data, axis=1)
    num_correct = np.sum(pred_class == y)
    return 100.0*(num_correct)/y.shape[0]

def measure_test_accuracy(model, x_test, y_test):
    chunk_size = 500
    test_accuracy = 0.0
    counter = 0.0
    for start in range(0, y_test.shape[0], chunk_size):
        xdata = x_test[start:min(start+chunk_size, x_test.shape[0])]
        ydata = y_test[start:min(start+chunk_size, y_test.shape[0])]
        if USE_GPU:
            xvar = Variable(torch.from_numpy(xdata).float().cuda(), requires_grad=False)
        else:
            xvar = Variable(torch.from_numpy(xdata).float(), requires_grad=False)
        yhat = model(xvar)
        test_accuracy += measure_accuracy(yhat, ydata)
        counter += 1
    return test_accuracy/counter

# ---------------------------

model = make_model(num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

for ep in range(epochs):
    rand_idx = np.random.permutation(x_train.shape[0])
    epoch_loss = 0.0
    train_accuracy = 0.0
    counter = 0.0
    ts = timer.time()
    for mb in range(int(x_train.shape[0]/batch_size)-2):
        xdata = x_train[rand_idx[mb*batch_size: (mb+1)*batch_size]]
        ydata = y_train[rand_idx[mb*batch_size: (mb+1)*batch_size]]
        batch_loss, batch_accuracy = train_step(model, optimizer, xdata, ydata)
        epoch_loss += batch_loss
        train_accuracy += batch_accuracy
        counter += 1

    test_accuracy = measure_test_accuracy(model, x_test, y_test)
    print("Epoch = %i | Train loss = %f | Train accuracy = %f | Test accuracy = %f | time = %f" %
          (ep+1, epoch_loss/counter, train_accuracy/counter, test_accuracy, timer.time()-ts))
