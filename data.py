import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

from cli import *
import vars as v
import args

def _get_mnist_dataloaders():
    v.num_classes = 10
    transforms_train = transforms.Compose([
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = datasets.MNIST(root=args.data_path, train=True, transform=transforms_train, download=True)
    mnist_test = datasets.MNIST(root=args.data_path, train=False, transform=transforms_test, download=False)
    mnist_train, mnist_valid = random_split(dataset=trainset, lengths=[0.8, 0.2])
    v.trainloader = DataLoader(dataset=mnist_train, shuffle=True, batch_size=64)
    v.validloader = DataLoader(dataset=mnist_valid, shuffle=True, batch_size=64)
    v.testloader = DataLoader(dataset=mnist_test, shuffle=True, batch_size=64)
    
def mnist():
    return _get_mnist_dataloaders()