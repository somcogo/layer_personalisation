import logging
import os
import requests
import shutil
import sys
import tarfile

import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from pathlib import Path
import h5py

class TruncatedDataset(Dataset):
    def __init__(self, dataset, dataset_name, indices=None):
        super().__init__()
        self.data = dataset.data
        if dataset_name == 'pascalvoc':
            self.labels = dataset.labels
        else:
            self.labels = dataset.targets

        if indices is not None:
            self.data = self.data[indices]
            self.labels = self.labels[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        return img, label

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
    dataset.targets = np.array(dataset.targets)
    val_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_pascal_voc_datasets(data_dir, use_hdf5=True):
    if use_hdf5:
        dataset = QuickPascalVocAugmentedSegmentation(data_dir=data_dir, mode='trn')
        val_dataset = QuickPascalVocAugmentedSegmentation(data_dir=data_dir, mode='val')
    else:
        dataset = PascalVocAugmentedSegmentation(root_dir=data_dir, split='train')
        val_dataset = PascalVocAugmentedSegmentation(root_dir=data_dir, split='val')
    return dataset, val_dataset

class QuickPascalVocAugmentedSegmentation(Dataset):
    def __init__(self, data_dir, mode, data_idxs=None):
        super().__init__()
        file_path = os.path.join(data_dir, 'aug_pascalvoc_{}.hdf5'.format(mode))
        hdf5_file = h5py.File(file_path, 'r')
        image_ds = hdf5_file['data']
        mask_ds = hdf5_file['mask']
        targets_ds = hdf5_file['targets']
        if data_idxs is not None:
            image_ds = image_ds[data_idxs]
            mask_ds = mask_ds[data_idxs]
            targets_ds = targets_ds[data_idxs]
        self.data = torch.from_numpy(np.array(image_ds))
        self.labels = torch.from_numpy(np.array(mask_ds))
        self.targets = np.array(targets_ds, dtype=object)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        image = self.data[index]
        mask = self.labels[index].to(dtype=torch.long)
        return (image, mask)

class PascalVocAugmentedSegmentation(Dataset):

    def __init__(self,
                 root_dir='../../data/pascal_voc_augmented',
                 split='train',
                 download_dataset=False,
                 transform=None,
                 data_idxs=None):
        """
        The dataset class for Pascal VOC Augmented Dataset.
        Args:
            root_dir: The path to the dataset.
            split: The type of dataset to use (train, test, val).
            download_dataset: Specify whether to download the dataset if not present.
            transform: The custom transformations to be applied to the dataset.
            data_idxs: The list of indexes used to partition the dataset.
        """
        self.root_dir = root_dir
        self.images_dir = Path('{}/dataset/img'.format(root_dir))
        self.masks_dir = Path('{}/dataset/cls'.format(root_dir))
        self.split_file = Path('{}/dataset/{}.txt'.format(root_dir, split))
        self.split = split
        self.transform = transform
        self.images = list()
        self.masks = list()
        self.targets = None

        if download_dataset:
            self.__download_dataset()

        self.__preprocess()
        if data_idxs is not None:
            self.images = [self.images[i] for i in data_idxs]
            self.masks = [self.masks[i] for i in data_idxs]

        self.__generate_targets()

    def __download_dataset(self):
        """
        Downloads the PASCAL VOC Augmented dataset.
        """
        files = {
            'pascalvocaug': {
                'name': 'PASCAL Train and Test Augmented Dataset',
                'file_path': Path('{}/benchmark.tgz'.format(self.root_dir)),
                'url': 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark'
                       '.tgz',
                'unit': 'GB'
            }
        }

        _download_file(**files['pascalvocaug'])
        _extract_file(files['pascalvocaug']['file_path'], self.root_dir)
        shutil.move('{}/benchmark_RELEASE/dataset'.format(self.root_dir), self.root_dir)
        shutil.rmtree('{}/benchmark_RELEASE'.format(self.root_dir))

    def __preprocess(self):
        """
        Pre-process the dataset to get mask and file paths of the images.
        Raises:
            AssertionError: When length of images and masks differs.
        """
        with open(self.split_file, 'r') as file_names:
            for file_name in file_names:
                img_path = Path('{}/{}.jpg'.format(self.images_dir, file_name.strip(' \n')))
                mask_path = Path('{}/{}.mat'.format(self.masks_dir, file_name.strip(' \n')))
                assert os.path.isfile(img_path)
                assert os.path.isfile(mask_path)
                self.images.append(img_path)
                self.masks.append(mask_path)
            assert len(self.images) == len(self.masks)

    def __generate_targets(self):
        """
        Used to generate targets which in turn is used to partition data in an non-IID setting.
        """
        targets = list()
        for i in range(len(self.images)):
            mat = sio.loadmat(self.masks[i], mat_dtype=True, squeeze_me=True, struct_as_record=False)
            categories = mat['GTcls'].CategoriesPresent
            if isinstance(categories, np.ndarray):
                categories = np.asarray(list(categories))
            else:
                categories = np.asarray([categories]).astype(np.uint8)
            targets.append(categories)
        self.targets = np.asarray(targets, dtype=object)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mat = sio.loadmat(self.masks[index], mat_dtype=True, squeeze_me=True, struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        mask = Image.fromarray(mask)

        if self.split == 'train':
            mean = [0.4585, 0.4389, 0.4058]
            std = [0.2664, 0.2634, 0.2774]
        elif self.split == 'val':
            mean = [0.4561, 0.4353, 0.4013]
            std = [0.2657, 0.2625, 0.2771]
        img_trans = Compose([Resize((224, 224)), ToTensor(), Normalize(mean, std)])
        mask_trans = Compose([Resize((224, 224), interpolation=InterpolationMode.NEAREST), ToTensor()])
        img = img_trans(img)
        mask = mask_trans(mask) * 255
        mask = mask.to(dtype=int)
        return (img, mask)

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """
        Returns:
            The clasess present in the Pascal VOC Augmented dataset.
        """
        return ('__background__', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'television',
                'train')
    
def __convert_size(size_in_bytes, unit):
    """
  Converts the bytes to human readable size format.
  Args:
    size_in_bytes (int): The number of bytes to convert
    unit (str): The unit to convert to.
  """
    if unit == 'GB':
        return '{:.2f} GB'.format(size_in_bytes / (1024 * 1024 * 1024))
    elif unit == 'MB':
        return '{:.2f} MB'.format(size_in_bytes / (1024 * 1024))
    elif unit == 'KB':
        return '{:.2f} KB'.format(size_in_bytes / 1024)
    else:
        return '{:.2f} bytes'.format(size_in_bytes)

def _download_file(name, url, file_path, unit):
    """
  Downloads the file to the path specified
  Args:
    name (str): The name to print in console while downloading.
    url (str): The url to download the file from.
    file_path (str): The local path where the file should be saved.
    unit (str): The unit to convert to.
  """
    with open(file_path, 'wb') as f:
        logging.info('Downloading {}...'.format(name))
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise EnvironmentError('Encountered error while fetching. Status Code: {}, Error: {}'.format(response.status_code,
                                                                                              response.content))
        total = response.headers.get('content-length')
        human_readable_total = __convert_size(int(total), unit)

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                human_readable_downloaded = __convert_size(int(downloaded), unit)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write(
                    '\r[{}{}] {}% ({}/{})'.format('#' * done, '.' * (50 - done), int((downloaded / total) * 100),
                                                  human_readable_downloaded, human_readable_total))
                sys.stdout.flush()
    sys.stdout.write('\n')
    logging.info('Download Completed.')


def _extract_file(file_path, extract_dir):
    """
  Extracts the file to the specified path.
  Args:
    file_path (str): The local path where the zip file is located.
    extract_dir (str): The local path where the files must be extracted.
  """
    with tarfile.open(file_path) as tar:
        logging.info('Extracting {} to {}...'.format(file_path, extract_dir))
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=extract_dir)
        tar.close()
        os.remove(file_path)
        logging.info('Extracted {}'.format(file_path))