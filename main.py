'''
Author: Yumo Chi
Modified based on the work from the neural_networks_tutorial.py from pytorch tutorial
as well as codes provided in CS598D's class notes.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

import torchvision
import torchvision.transforms as transforms
# import torchsample as ts

import numpy as np
import h5py
from random import randint
from torch.autograd import Variable

from torchvision import datasets
from torch.utils.data import DataLoader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # kernel
        # 1 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=2)

        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
       
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
       
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
               
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
       
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 64 input image channel, 64 output channels, 4x4 square convolution, 1 stride, padding 2 
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # batch normalization
        # normalize conv1, conv3, conv5, conv7, conv8 64 output channels
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8_bn = nn.BatchNorm2d(64)

        #drop out layers for conv2, conv4, conv6, conv8
        self.conv2_dol = nn.Dropout(p=0.1)
        self.conv4_dol = nn.Dropout(p=0.1)
        self.conv6_dol = nn.Dropout(p=0.1)
        self.conv8_dol = nn.Dropout(p=0.1)

        # 2 fully connected layer 
        self.fc1 = nn.Linear(1024, 100)
        self.fc2 = nn.Linear(100, 10)

    def num_flat_features(self, x):
        size = x[0].size() # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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
        # apply first fully connected layer
        x = F.relu(self.fc1(x))
        # apply 2nd fully connected layer
        x = self.fc2(x)
        # apply soft_max to the result
        return F.log_softmax(x, dim=1)



def main():
    # set up argparser
    parser = argparse.ArgumentParser(description='hw3')
    # batch-size
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    # epochs
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    # learning rate
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    # monte carlo sample times 
    parser.add_argument('--mck', type=int, default=50, metavar='K',
                        help='number of network sampled for monte carlo (default: 50)')

    # gpu setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    args = parser.parse_args()

    # test if gpu should be used
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # CIFAR10_data = h5py.File('CIFAR10.hdf5', 'r')
    # x_train = np.float32(CIFAR10_data['X_train'][:])

    # y_train = np.int32(np.array(CIFAR10_data['Y_train']))

    # # x_test = np.float32(CIFAR10_data['X_test'][:])
    # x_test = np.float32(CIFAR10_data['X_test'][:] )
    # y_test = np.int32( np.array(CIFAR10_data['Y_test']))

    # adding in data augmentation transformations
    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # just transform to tensor for test_data
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # data loader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transformations)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=1)
    batch_size = args.batch_size


    model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs
    # L_Y_train = len(y_train)
    model.train()
    # train_loss = []

    for epoch in range(1, args.epochs + 1):
        #Randomly shuffle data every epoch
        # L_Y_train = len(y_train)
        # L_Y_train = 10000
        # I_permutation = np.random.permutation(L_Y_train)

        # x_train = x_train[I_permutation,:]

        # y_train = y_train[I_permutation]
        train_accu = []
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data), Variable(target)

            data, target = data.to(device), target.to(device)
            # x_train_batch = torch.FloatTensor( x_train[i:i+batch_size,:] )
            # y_train_batch = torch.LongTensor( y_train[i:i+batch_size] )
            # data, target = Variable(x_train_batch), Variable(y_train_batch)
            # data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)

            loss = F.nll_loss(output, target)

            loss.backward()

            # train_loss.append(loss.data[0])

            optimizer.step()
            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)
               )*100.0
            train_accu.append(accuracy)

        accuracy_epoch = np.mean(train_accu)
        print(epoch, accuracy_epoch)
    
    model.eval()
    test_accu = []
    #L_Y_test = len(y_test)
    # for i in range(0, L_Y_test, batch_size):
        # x_test_batch = torch.FloatTensor( x_test[i:i+batch_size,:] )
        # y_test_batch = torch.LongTensor( y_test[i:i+batch_size] )
        # data, target = Variable(x_test_batch), Variable(y_test_batch)
        # data, target = data.to(device), target.to(device)
    for batch_idx, (data, target) in enumerate(test_loader, 0):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)
           )*100.0
        test_accu.append(accuracy)
    accuracy_test = np.mean(test_accu)
    print('test without activating dropout')
    print(accuracy_test)

    k = args.mck
    # perform monte carlo step
    # activate dropout layers
    model.train()
    mc_accu = [[]] * k
    test_accu = []

    for batch_idx, (data, target) in enumerate(test_loader, 0):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        for i in range(k):
            output = model(data)
            if i == 0:   
                mc_prediction = output.data.max(1)[1] # first column has actual prob.
            else:
                mc_prediction += output.data.max(1)[1] # first column has actual prob.

            mc_prediction = torch.div(mc_prediction, i+1)
            mc_accuracy = ( float( mc_prediction.eq(target.data).sum() ) /float(batch_size)
           )*100.0
            mc_accu[i].append(mc_accuracy)

        output = torch.div(output, k)

        loss = F.nll_loss(output, target)
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)
           )*100.0
        test_accu.append(accuracy)

    # print('test with activated dropout')
    # accuracy_test = np.mean(test_accu)
    # print(accuracy_test)

    # print(mc_accu)

    print('showing result with different sample size')
    for i in range(k):
        print('{} samples : {}'. format(batch_size* (i+1), np.mean(mc_accu[i])))

if __name__ == '__main__':
    main()