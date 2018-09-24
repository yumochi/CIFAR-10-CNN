'''
Author: Yumo Chi
Modified based on the work from the neural_networks_tutorial.py from pytorch tutorial
'''
# -*- coding: utf-8 -*-
"""
Neural Networks
===============

Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

For example, look at this network that classifies digit images:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``

Define the network
------------------

Let’s define this network:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

import torchvision
import torchvision.transforms as transforms

import numpy as np
import h5py
from random import randint

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        '''
            class torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, 
            bias=True)
                        '''
        # kernel
        # 1 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=2)
        print('conv1')
        # print(self.conv1.size())
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        # print(self.conv2.size())
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        # print(self.conv3.size())
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        # print(self.conv4.size())        
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        # print(self.conv5.size())
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # batch normalization
        # normalize conv1 64 output channels
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8_bn = nn.BatchNorm2d(64)

        #drop out
        self.conv2_dol = nn.Dropout(p=0.2)
        self.conv4_dol = nn.Dropout(p=0.2)
        self.conv6_dol = nn.Dropout(p=0.2)
        self.conv8_dol = nn.Dropout(p=0.2)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64 * 8 * 8, 500)
        self.fc2 = nn.Linear(500, 10)

    def num_flat_features(self, x):
        size = x.size() # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        ###################################Need to Change##########################################

    def forward(self, x):
        # apply batch normalization after applying 1st conv
        x = F.relu(self.conv1_bn(self.conv1(x)))
        # apply dropout max_pool after applying 2nd conv
        # Max pooling over a (2, 2) winow
        # If the size is a square you can only specify a single number
        x = self.conv2_dol(F.max_pool2d(F.relu(self.conv2(x)), (2, 2)))
        # apply batch normalization after applying 3rd conv
        x = F.relu(self.conv3_bn(self.conv3(x)))
        # apply dropout max_pool after applying 4th conv
        x = self.conv4_dol(F.max_pool2d(F.relu(self.conv4(x)), (2, 2)))
        # apply batch normalization max_pool after applying 5th conv
        x = F.relu(self.conv5_bn(self.conv5(x)))
        # apply dropout after applying 6th conv
        x = self.conv6_dol(F.relu(self.conv6(x)))
        # apply batch normalization after applying 7th conv
        x = self.conv7_bn(F.relu(self.conv7(x)))
        # apply dropout batch normalization after applying 8th conv
        x = self.conv8_dol(self.conv8_bn(F.relu(self.conv8(x))))
        # change to one dim colum vector
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.log_softmax(x, dim=1)


def train(args, model, device, x, y, optimizer, epoch):
    model.train()
    batch_size = len(x)

    y = torch.LongTensor((y))

    for batch_idx in range(batch_size):
        # for n in range( len(x_train)):
        n_random = randint(0,len(x)-1 )
        target = y[n_random]
        data = x[n_random][:]
        data = torch.FloatTensor(np.array([data] * 64))
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), batch_size,
                100. * batch_idx / batch_size, loss.item()))

def test(args, model, device, x, y):
    model.eval()
    test_loss = 0
    correct = 0
    batch_size = len(x)
    x = torch.FloatTensor((x))
    y = torch.LongTensor((y))

    with torch.no_grad():
        for i in range(batch_size):
            data, target = x[i], y[i]
            data, target = data.to(device), target.to(device)
            output = model(data)
            # use negative log loss
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= batch_size

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, batch_size,
        100. * correct / batch_size))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='hw3')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    ########################################################################
    # Load input from CIFAR-10

    CIFAR10_data = h5py.File('CIFAR10.hdf5', 'r')
    x_train = np.float32(CIFAR10_data['X_train'][:])
    # # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    y_train = np.int32(np.array(CIFAR10_data['Y_train']))
    # x_test = np.float32(CIFAR10_data['X_test'][:])
    x_test = np.float32(CIFAR10_data['X_test'][:] )
    y_test = np.int32( np.array(CIFAR10_data['Y_test']))


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, x_train, y_train, optimizer, epoch)
        test(args, model, device, x_test, y_test)


if __name__ == '__main__':
    main()