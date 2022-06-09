import torch
import torch.utils.data as data
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import copy
import time
import random
import numpy as np

def set_seed():
  SEED = 1234

  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True

def load_CIFAR_data(BATCH_SIZE = 256, ROOT = '.data', VALID_RATIO = 0.9):  
  train_data = datasets.CIFAR10(root=ROOT,
                                train=True,
                                download=True)
  means = train_data.data.mean(axis=(0, 1, 2)) / 255
  stds = train_data.data.std(axis=(0, 1, 2)) / 255

  train_transforms = transforms.Compose([
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(32, padding=2),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=means,
                                                std=stds)
                       ])

  test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=means,
                                                  std=stds)
                        ])

  train_data = datasets.CIFAR10(ROOT,
                                train=True,
                                download=True,
                                transform=train_transforms)

  test_data = datasets.CIFAR10(ROOT,
                              train=False,
                              download=True,
                              transform=test_transforms)

  n_train_examples = int(len(train_data) * VALID_RATIO)
  n_valid_examples = len(train_data) - n_train_examples

  train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
  valid_data = copy.deepcopy(valid_data)
  valid_data.dataset.transform = test_transforms

  train_iterator = data.DataLoader(train_data,
                                  shuffle=True,
                                  batch_size=BATCH_SIZE)

  valid_iterator = data.DataLoader(valid_data,
                                  batch_size=BATCH_SIZE)

  test_iterator = data.DataLoader(test_data,
                                  batch_size=BATCH_SIZE)

  return train_iterator, valid_iterator, test_iterator

def load_MNIST_data(BATCH_SIZE = 256, ROOT = '.data', VALID_RATIO = 0.9):  
  train_data = datasets.MNIST(root=ROOT,
                                train=True,
                                download=True)
  means = train_data.data.float().mean() / 255
  stds = train_data.data.float().std() / 255

  train_transforms = transforms.Compose([
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(32, padding=2),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=means,
                                                std=stds)
                       ])

  test_transforms = transforms.Compose([
                            transforms.Resize(32),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=means,
                                                  std=stds)
                        ])

  train_data = datasets.MNIST(ROOT,
                                train=True,
                                download=True,
                                transform=train_transforms)

  test_data = datasets.MNIST(ROOT,
                              train=False,
                              download=True,
                              transform=test_transforms)

  n_train_examples = int(len(train_data) * VALID_RATIO)
  n_valid_examples = len(train_data) - n_train_examples

  train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
  valid_data = copy.deepcopy(valid_data)
  valid_data.dataset.transform = test_transforms

  train_iterator = data.DataLoader(train_data,
                                  shuffle=True,
                                  batch_size=BATCH_SIZE)

  valid_iterator = data.DataLoader(valid_data,
                                  batch_size=BATCH_SIZE)

  test_iterator = data.DataLoader(test_data,
                                  batch_size=BATCH_SIZE)


  return train_iterator, valid_iterator, test_iterator

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
