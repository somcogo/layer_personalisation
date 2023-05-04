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

def partition_data(dataset_name, partition, n_sites):

    if dataset_name == 'cifar10':
        cifar10_train_ds, cifar10_test_ds = get_cifar10_datasets()
        X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
        X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    if partition == 'equal':
        if dataset_name == "cifar10":
            num = 2
            K = 10
        elif dataset_name == "cifar100":
            num = 10
            K = 100

        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
        assert (num * n_sites) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_sites) // K
        class_dict = {}
        for i in range(K):
            # sampling alpha_i_c
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
        class_partitions = defaultdict(list)
        for i in range(n_sites):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
        net_dataidx_map_train ={i:np.ndarray(0,dtype=np.int64) for i in range(n_sites)}
        net_dataidx_map_test ={i:np.ndarray(0,dtype=np.int64) for i in range(n_sites)}

        for usr_i in range(n_sites):
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i], data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i], data_class_idx_test[c][:end_idx_test])

                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]

    return (X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test)