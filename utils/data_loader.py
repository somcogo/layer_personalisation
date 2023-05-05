from collections import defaultdict
import random

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from .datasets import get_cifar10_datasets, get_cifar100_datasets, get_pascal_voc_datasets, TruncatedDataset
from .partition import partition_by_class, partition_with_dirichlet_distribution

data_path = 'data/'

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

def get_datasets(data_dir, dataset, use_hdf5=None):
    if dataset == 'cifar10':
        trn_dataset, val_dataset = get_cifar10_datasets(data_dir=data_dir)
    elif dataset == 'cifar100':
        trn_dataset, val_dataset = get_cifar100_datasets(data_dir=data_dir)
    elif dataset == 'pascalvoc':
        trn_dataset, val_dataset = get_pascal_voc_datasets(data_dir=data_dir, use_hdf5=use_hdf5)
    return trn_dataset, val_dataset

def get_dl_lists(dataset, partition, n_site, batch_size, alpha=None, use_hdf5=True):
    trn_dataset, val_dataset = get_datasets(data_dir=data_path, dataset=dataset, use_hdf5=use_hdf5)

    if partition == 'regular':
        trn_ds_list = [trn_dataset]
        val_ds_list = [val_dataset]
    elif partition == 'by_class':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_by_class(data_dir=data_path, dataset=dataset, n_sites=n_site)
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [TruncatedDataset(val_dataset, dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
    elif partition == 'dirichlet':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_with_dirichlet_distribution(data_dir=data_path, dataset=dataset, n_sites=n_site, alpha=alpha)
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [TruncatedDataset(val_dataset, dataset, idx_map) for idx_map in net_dataidx_map_test.values()]

    trn_dl_list = [DataLoader(dataset=trn_ds, batch_size=batch_size, shuffle=True, drop_last=False) for trn_ds in trn_ds_list]
    val_dl_list = [DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, drop_last=False) for val_ds in val_ds_list]
    return trn_dl_list, val_dl_list
