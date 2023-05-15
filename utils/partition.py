from collections import defaultdict
import random

import numpy as np

from .datasets import get_cifar10_datasets, get_cifar100_datasets, get_pascal_voc_datasets

def partition(data_dir, dataset, partition, n_sites, alpha=None):

    if partition == 'by-class':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_by_class(data_dir, dataset, n_sites)
    elif partition == 'dirichlet':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_with_dirichlet_distribution(data_dir, dataset, n_sites, alpha)
    return (net_dataidx_map_train, net_dataidx_map_test)

def partition_by_class(data_dir, dataset, n_sites):
    if dataset == 'cifar10':
        train_ds, test_ds = get_cifar10_datasets(data_dir)
    if dataset == 'cifar100':
        train_ds, test_ds = get_cifar100_datasets(data_dir)
    if dataset == 'pascalvoc':
        train_ds, test_ds = get_pascal_voc_datasets(data_dir, use_hdf5=True)
    y_train = train_ds.targets
    y_test = test_ds.targets

    if dataset == "cifar10":
        K = 10
        num = K // n_sites
    elif dataset == "cifar100":
        K = 100
        num = K // n_sites
    elif dataset == 'pascalvoc':
        K = 20
        num = K // n_sites

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
    if dataset == 'pascalvoc':
        data_class_idx_train = {i: np.where([i in img_labels for img_labels in y_train])[0] for i in range(K)}
        data_class_idx_test = {i: np.where([i in img_labels for img_labels in y_test])[0] for i in range(K)}
    else:
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

    return (net_dataidx_map_train, net_dataidx_map_test)

# TODO: link fedml, fedtp 
# https://github.com/zhyczy/FedTP/blob/main/utils.py
# https://github.com/FedML-AI/FedML/blob/master/python/fedml/core/data/noniid_partition.py

def partition_with_dirichlet_distribution(data_dir, dataset, n_sites, alpha):
    if dataset == 'cifar10':
        train_ds, test_ds = get_cifar10_datasets(data_dir)
    if dataset == 'cifar100':
        train_ds, test_ds = get_cifar100_datasets(data_dir)
    if dataset == 'pascalvoc':
        train_ds, test_ds = get_pascal_voc_datasets(data_dir, use_hdf5=True)
    y_train = train_ds.targets
    y_test = test_ds.targets

    min_size = 0
    min_require_size = 10
    if dataset == 'cifar10':
        K = 10
    elif dataset == 'cifar100':
        K = 100
    elif dataset == 'pascalvoc':
        K = 20

    N_train = len(y_train)
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_sites)]
        idx_batch_test = [[] for _ in range(n_sites)]
        for k in range(K):
            if dataset == 'pascalvoc':
                if k == 0:
                    train_idx_k = np.asarray([np.any(y_train[i] == k) for i in range(len(y_train))])
                    test_idx_k = np.asarray([np.any(y_test[i] == k) for i in range(len(y_test))])
                else:
                    train_idx_k = np.asarray([np.any(y_train[i] == k)
                                                and not np.any(np.in1d(y_train[i], range(k - 1)))
                                                for i in range(len(y_train))])
                    test_idx_k = np.asarray([np.any(y_test[i] == k)
                                                and not np.any(np.in1d(y_test[i], range(k - 1)))
                                                for i in range(len(y_test))])
                train_idx_k = np.where(train_idx_k)[0]
                test_idx_k = np.where(test_idx_k)[0]
            else:
                train_idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]

            idx_batch_train, idx_batch_test, min_size = partition_class_samples_with_dirichlet_distribution(N_train, alpha, n_sites, idx_batch_train, idx_batch_test, train_idx_k, test_idx_k)
        
        for j in range(n_sites):
            np.random.shuffle(idx_batch_train[j])
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_train[j] = idx_batch_train[j]
            net_dataidx_map_test[j] = idx_batch_test[j]

    return (net_dataidx_map_train, net_dataidx_map_test)

def partition_class_samples_with_dirichlet_distribution(N, alpha, n_sites, idx_batch_train, idx_batch_test, train_idx_k, test_idx_k):

    np.random.shuffle(train_idx_k)
    np.random.shuffle(test_idx_k)
    
    proportions = np.random.dirichlet(np.repeat(alpha, n_sites))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / n_sites) for p, idx_j in zip(proportions, idx_batch_train)]
    )
    proportions = proportions / proportions.sum()
    proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
    proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]

    idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(train_idx_k, proportions_train))]
    idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(test_idx_k, proportions_test))]

    min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
    min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
    min_size = min(min_size_train, min_size_test)

    return idx_batch_train, idx_batch_test, min_size
