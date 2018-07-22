'''Trains a simple convnet on the MNIST dataset.
Gets to 99.2% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
4 seconds per epoch on a GTX 1080 Ti.
'''
from os import environ
# Select GPU 0 only
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['CUDA_VISIBLE_DEVICES']='0'
environ['MKL_THREADING_LAYER']='GNU'

from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
from keras.datasets import mnist
from tqdm import tqdm
import argparse
import torch
import keras
import torch.nn as nn
import numpy as np
import time as timer
import torch.nn.functional as F
import torch.optim as optim


# hyperparameters
# ---------------------------


USE_GPU = True
batch_size = 64
num_classes = 10
epochs = 12


# data loading
# ---------------------------


img_rows, img_cols = 28, 28
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


def make_data_loader(X, Y, batch_size, shuffle=True, use_gpu=False):
    if use_gpu:
        dataset = TensorDataset(torch.FloatTensor(X).cuda(), torch.LongTensor(Y).cuda())
    else:
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(Y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


# Helper function
# ---------------------------


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


def to_cuda(input):
    if USE_GPU:
        return input.cuda()
    else:
        return input


def to_numpy(input):
    if USE_GPU:
        try:
            return input.cpu().data.numpy()
        except:
            return input.cpu().numpy()
    else:
        try:
            return input.data.numpy()
        except:
            return input.numpy()


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
    return to_cuda(model)


def train_step(model, optimizer, x_batch, y_batch, loss_function):
    model.train()
    xvar, yvar = Variable(x_batch), Variable(y_batch)
    optimizer.zero_grad()
    yhat = model(xvar)
    loss = loss_function(yhat, yvar)
    batch_loss = to_numpy(loss)
    batch_accuracy = measure_accuracy(yhat, y_batch)
    loss.backward()
    optimizer.step()
    return batch_loss, batch_accuracy


def measure_accuracy(yhat, y):
    yhat_data = to_numpy(yhat)
    ydata = to_numpy(y).ravel()
    # find class with largest score
    pred_class = np.argmax(yhat_data, axis=1)
    num_correct = np.sum(pred_class == ydata)
    return 100.0*(num_correct)/y.shape[0]


def measure_test_accuracy(model, x_test, y_test):
    model.eval()
    chunk_size = 500
    counter = 0.0
    num_correct = 0
    for start in range(0, y_test.shape[0], chunk_size):
        xdata = x_test[start:min(start+chunk_size, x_test.shape[0])]
        ydata = y_test[start:min(start+chunk_size, y_test.shape[0])]
        xvar = Variable(to_cuda(torch.FloatTensor(xdata)))
        yhat = model(xvar)
        yhat_data = to_numpy(yhat)
        pred_class = np.argmax(yhat_data, axis=1)
        num_correct += np.sum(pred_class == ydata)
        counter += ydata.shape[0]
    return 100.0 * num_correct / counter


# ---------------------------


model = make_model(num_classes)
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

data_loader = make_data_loader(x_train, y_train, batch_size=batch_size, shuffle=True, use_gpu=USE_GPU)

for ep in range(epochs):
    epoch_loss = 0.0
    train_accuracy = 0.0
    counter = 0.0
    ts = timer.time()
    for mb, (x_batch, y_batch) in enumerate(data_loader):
        x_batch, y_batch = to_cuda(x_batch), to_cuda(y_batch)
        batch_loss, batch_accuracy = train_step(model, optimizer, x_batch, y_batch, loss_function)
        epoch_loss += batch_loss
        train_accuracy += batch_accuracy
        counter += 1

    test_accuracy = measure_test_accuracy(model, x_test, y_test)
    print("Epoch = %i | Train loss = %f | Train accuracy = %f | Test accuracy = %f | time = %f" %
          (ep+1, epoch_loss/counter, train_accuracy/counter, test_accuracy, timer.time()-ts))
