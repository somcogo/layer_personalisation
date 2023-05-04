import time

import h5py
import numpy as np

from utils.datasets import PascalVocAugmentedSegmentation

t1 = time.time()

def preprocess_pascal_voc():
    trn_dataset = PascalVocAugmentedSegmentation(root_dir='data', split='train')
    val_dataset = PascalVocAugmentedSegmentation(root_dir='data', split='val')
    trn_number = len(trn_dataset)
    val_number = len(val_dataset)

    special_dtype = h5py.vlen_dtype(np.dtype('int'))

    f_trn = h5py.File('data/aug_pascalvoc_trn.hdf5', 'w')
    f_trn.create_dataset('data', shape=(trn_number, 3, 224, 224), chunks=(1, 3, 224, 224), dtype='float32')
    f_trn.create_dataset('mask', shape=(trn_number, 1, 224, 224), chunks=(1, 1, 224, 224), dtype='float32')
    f_trn.create_dataset('targets', data=trn_dataset.targets, dtype=special_dtype)

    f_val = h5py.File('data/aug_pascalvoc_val.hdf5', 'w')
    f_val.create_dataset('data', shape=(val_number, 3, 224, 224), chunks=(1, 3, 224, 224), dtype='float32')
    f_val.create_dataset('mask', shape=(val_number, 1, 224, 224), chunks=(1, 1, 224, 224), dtype='float32')
    f_val.create_dataset('targets', data=val_dataset.targets, dtype=special_dtype)

    t2 = time.time()
    print('created files', t2-t1)
    for ndx, image_tuple in enumerate(val_dataset):
        img, mask = image_tuple
        f_val['data'][ndx] = img
        f_val['mask'][ndx] = mask
        if ndx % 100 == 0:
            f_val.flush()
    f_val.close()

    t3 = time.time()
    print('saved validation imgs', t3-t2)
    for ndx, image_tuple in enumerate(trn_dataset):
        img, mask = image_tuple
        f_trn['data'][ndx] = img
        f_trn['mask'][ndx] = mask
        if ndx % 100 == 0:
            f_trn.flush()
    f_trn.close()
    t4 = time.time()
    print('saved training imgs', t4-t3)