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

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import time as timer

DATA_DIR = '~/data/'

# hyperparameters
# ---------------------------


USE_GPU = True
batch_size = 64
num_classes = 10
epochs = 12


# data loading
# ---------------------------

device = torch.device("cuda" if USE_GPU else "cpu")
kwargs = {'num_workers': 6, 'pin_memory': True} if USE_GPU else {}

train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)


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


def number_correct(yhat, y):
    yhat_data = to_numpy(yhat)
    ydata = to_numpy(y).ravel()
    # find class with largest score
    pred_class = np.argmax(yhat_data, axis=1)
    num_correct = np.sum(pred_class == ydata)
    return num_correct


def measure_accuracy(yhat, y):
    num_correct = number_correct(yhat, y)
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

for ep in range(epochs):
    epoch_loss = 0.0
    train_accuracy = 0.0
    counter = 0.0
    ts = timer.time()
    for idx, mini_batch in enumerate(tqdm(train_dataloader)):
        x_batch, y_batch = mini_batch[0], mini_batch[1].view(-1)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        batch_loss, batch_accuracy = train_step(model, optimizer, x_batch, y_batch, loss_function)
        epoch_loss += batch_loss
        train_accuracy += batch_accuracy
        counter += 1
    train_accuracy = train_accuracy/counter

    # measure test performance per epoch
    num_correct = 0
    counter = 0
    model.eval()
    for idx, mini_batch in enumerate(test_dataloader):
        x_batch, y_batch = mini_batch[0], mini_batch[1].view(-1)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_hat = model(Variable(x_batch))
        num_correct += number_correct(y_hat, y_batch)
        counter += y_batch.shape[0]
    test_accuracy = 100.0 * num_correct / counter
    print("Epoch = %i | Train loss = %f | Train accuracy = %f | Test accuracy = %f | time = %f" %
          (ep+1, epoch_loss/counter, train_accuracy, test_accuracy, timer.time()-ts))
