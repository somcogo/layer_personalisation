from collections import defaultdict
import random

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from .datasets import PascalVocAugmentedSegmentation, QuickPascalVocAugmentedSegmentation

data_path = 'data/'

def get_cifar10_datasets(data_dir):
    train_mean = [0.4914, 0.4822, 0.4465]
    train_std = [0.2470, 0.2435, 0.2616]
    train_transform = Compose([ToTensor(), Normalize(train_mean, train_std)])
    val_mean = [0.4942, 0.4851, 0.4504]
    val_std = [0.2467, 0.2429, 0.2616]
    val_transform = Compose([ToTensor(), Normalize(val_mean, val_std)])

    dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    dataset.targets = np.array(dataset.targets)
    val_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_cifar100_datasets(data_dir):
    train_mean = [0.5071, 0.4866, 0.4409]
    train_std = [0.2673, 0.2564, 0.2762]
    train_transform = Compose([ToTensor(), Normalize(train_mean, train_std)])
    val_mean = [0.5088, 0.4874, 0.4419]
    val_std = [0.2683, 0.2574, 0.2771]
    val_transform = Compose([ToTensor(), Normalize(val_mean, val_std)])

    dataset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)

    return dataset, val_dataset

def get_pascal_voc_datasets(data_dir, use_hdf5=True):
    if use_hdf5:
        dataset = QuickPascalVocAugmentedSegmentation(data_dir=data_dir, mode='trn')
        val_dataset = QuickPascalVocAugmentedSegmentation(data_dir=data_dir, mode='val')
    else:
        dataset = PascalVocAugmentedSegmentation(root_dir=data_dir, split='train')
        val_dataset = PascalVocAugmentedSegmentation(root_dir=data_dir, split='val')
    return dataset, val_dataset

def get_cifar10_dl(partition, n_sites, batch_size):
    if partition == 'regular':
        dataset, val_dataset = get_cifar10_datasets(data_path)

    train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl

def get_cifar100_dl(partition, n_sites, batch_size):
    if partition == 'regular':
        dataset, val_dataset = get_cifar100_datasets(data_path)

    train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl

def get_pascal_voc_dl(partition, n_site, batch_size, use_hdf5=True):
    print('using hdf5:', use_hdf5)
    if partition == 'regular':
        dataset, val_dataset = get_pascal_voc_datasets(data_path, use_hdf5=use_hdf5)
    
    train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl
