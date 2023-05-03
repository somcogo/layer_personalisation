import random

import numpy as np
import torch
from torchvision.transforms import (
     functional,
     RandomResizedCrop,
     Compose,
     Pad,
     RandomCrop,
     RandomHorizontalFlip,
     RandomApply,
     RandomRotation,
     RandomErasing)

def aug_image(batch: torch.Tensor, mask_batch, mode):
    if mode == 'segmentation':
        batch = aug_flip_rotate_scale(batch, mask_batch)
    elif mode == 'classification':
        batch = aug_crop_rotate_flip_erase(batch)
    return batch, mask_batch

def aug_flip_rotate_scale(batch, mask_batch):
    flip = random.choice([True, False])
    angle = random.choice([0, 90, 180, 270])
    scale = random.uniform(0.9, 1.1)
    if flip:
        batch = functional.hflip(batch)
        mask_batch = functional.hflip(mask_batch)
    batch = functional.rotate(batch, angle)
    mask_batch = functional.rotate(mask_batch, angle)
    batch = scale * batch
    return batch

def aug_crop_rotate_flip_erase(batch):
    trans = Compose([
        Pad(4),
        RandomCrop(32),
        RandomHorizontalFlip(p=0.25),
        RandomApply(torch.nn.ModuleList([
            RandomRotation(degrees=15)
        ]), p=0.25),
        RandomErasing(p=0.5, scale=(0.015625, 0.25), ratio=(0.25, 4))
    ])
    batch = trans(batch)
    return batch

def batch_miou(batch_pred: torch.Tensor, batch_mask: torch.Tensor):
    assert batch_pred.shape == batch_mask.shape
    original_shape = batch_pred.shape
    h, w = original_shape[-2:]
    batch_pred = batch_pred.view(-1, h, w)
    batch_mask = batch_mask.view(-1, h, w)
    batch_miou = torch.zeros(batch_pred.shape[0])
    for i in range(batch_pred.shape[0]):
        batch_miou[i] = miou(batch_pred[i], batch_mask[i])

    return batch_miou.view(original_shape[:-2])


def miou(pred: torch.Tensor, mask: torch.Tensor):
    classes = pred.unique()

    iou_sum= 0
    for obj_class in classes:
        class_pred = pred == obj_class
        class_mask = mask == obj_class
        iou_sum += iou(class_pred, class_mask)
    mean_iou = iou_sum / len(classes)
    return mean_iou

def iou(class_pred, class_mask):
    intersection = torch.sum(torch.logical_and(class_pred, class_mask))
    union = torch.sum(torch.logical_or(class_pred, class_mask))

    if union == 0:
        return 0
    else:
        return intersection / union