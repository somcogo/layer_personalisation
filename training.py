import sys
import argparse
import os
import datetime
import shutil
import random

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import torchvision

from models.model import ResNet18Model, ResNet34Model, TinySwin, SmallSwin, LargeSwin, UnetWithResNet34, ResNetWithEmbeddings
from utils.logconf import logging
from utils.data_loader import get_dl_lists
from utils.ops import aug_image, batch_miou
from utils.merge_strategies import get_layer_list

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class LayerPersonalisationTrainingApp:
    def __init__(self, sys_argv=None, epochs=None, batch_size=None, logdir=None, lr=None, comment=None, dataset='cifar10', site_number=5, model_name=None, optimizer_type=None, scheduler_mode=None, label_smoothing=None, T_max=None, pretrained=None, aug_mode=None, save_model=None, partition=None, alpha=None, background_weight=None, strategy=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser(description="Test training")
        parser.add_argument("--epochs", default=2, type=int, help="number of training epochs")
        parser.add_argument("--batch_size", default=500, type=int, help="number of batch size")
        parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
        parser.add_argument("--in_channels", default=3, type=int, help="number of image channels")
        parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
        parser.add_argument("--dataset", default='cifar10', type=str, help="dataset to train on")
        parser.add_argument("--model_name", default='resnet34', type=str, help="name of the model to use")
        parser.add_argument("--optimizer_type", default='adam', type=str, help="type of optimizer to use")
        parser.add_argument("--label_smoothing", default=0.0, type=float, help="label smoothing in Cross Entropy Loss")
        parser.add_argument("--T_max", default=1000, type=int, help="T_max in Cosine LR scheduler")
        parser.add_argument("--pretrained", default=False, type=bool, help="use pretrained model")
        parser.add_argument("--aug_mode", default='segmentation', type=str, help="mode of data augmentation")
        parser.add_argument("--scheduler_mode", default=None, type=str, help="choice of LR scheduler")
        parser.add_argument("--save_model", default=False, type=bool, help="save models during training")
        parser.add_argument("--partition", default='regular', type=str, help="how to partition the data among sites")
        parser.add_argument("--alpha", default=None, type=float, help="alpha used for the Dirichlet distribution")
        parser.add_argument("--background_weight", default=1, type=float, help="weight of background in XE loss")
        parser.add_argument("--strategy", default='all', type=str, help="merging strategy")
        parser.add_argument('comment', help="Comment suffix for Tensorboard run.", nargs='?', default='dwlpt')

        self.args = parser.parse_args()
        if epochs is not None:
            self.args.epochs = epochs
        if batch_size is not None:
            self.args.batch_size = batch_size
        if logdir is not None:
            self.args.logdir = logdir
        if lr is not None:
            self.args.lr = lr
        if comment is not None:
            self.args.comment = comment
        if dataset is not None:
            self.args.dataset = dataset
        if site_number is not None:
            self.args.site_number = site_number
        if model_name is not None:
            self.args.model_name = model_name
        if optimizer_type is not None:
            self.args.optimizer_type = optimizer_type
        if label_smoothing is not None:
            self.args.label_smoothing = label_smoothing
        if T_max is not None:
            self.args.T_max = T_max
        if pretrained is not None:
            self.args.pretrained = pretrained
        if aug_mode is not None:
            self.args.aug_mode = aug_mode
        if scheduler_mode is not None:
            self.args.scheduler_mode = scheduler_mode
        if save_model is not None:
            self.args.save_model = save_model
        if partition is not None:
            self.args.partition = partition
        if alpha is not None:
            self.args.alpha = alpha
        if background_weight is not None:
            self.args.background_weight = background_weight
        if strategy is not None:
            self.args.strategy = strategy
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.args.logdir)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.models = self.initModels()
        self.optims = self.initOptimizers()
        self.schedulers = self.initSchedulers()

    def initModels(self):
        if self.args.dataset == 'cifar10':
            num_classes = 10
        elif self.args.dataset == 'cifar100':
            num_classes = 100
        elif self.args.dataset == 'pascalvoc':
            num_classes = 21
        models = []
        for _ in range(self.args.site_number):
            if self.args.model_name == 'resnet18':
                model = ResNet18Model(num_classes=num_classes, pretrained=self.args.pretrained)
            if self.args.model_name == 'resnet34':
                model = ResNet34Model(num_classes=num_classes, pretrained=self.args.pretrained)
            elif self.args.model_name == 'swint':
                model = TinySwin(num_classes=num_classes, pretrained=self.args.pretrained)
            elif self.args.model_name == 'swins':
                model = SmallSwin(num_classes=num_classes, pretrained=self.args.pretrained)
            elif self.args.model_name == 'swinl':
                model = LargeSwin(num_classes=num_classes, pretrained=self.args.pretrained)
            elif self.args.model_name == 'unetresnet34':
                model = UnetWithResNet34(num_classes=num_classes, pretrained=self.args.pretrained)
            elif self.args.model_name == 'resnet34emb':
                model = ResNetWithEmbeddings(num_classes=num_classes)
            models.append(model)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            for model in models:
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(self.device)
        return models

    def initOptimizers(self):
        optims = []
        for model in self.models:
            if self.args.optimizer_type == 'adam':
                optim = Adam(params=model.parameters(), lr=self.args.lr)
            elif self.args.optimizer_type == 'adamw':
                optim = AdamW(params=model.parameters(), lr=self.args.lr, weight_decay=0.05)
            elif self.args.optimizer_type == 'sgd':
                optim = SGD(params=model.parameters(), lr=self.args.lr, weight_decay=0.0001, momentum=0.9)
            optims.append(optim)
        return optims
    
    def initSchedulers(self):
        if self.args.scheduler_mode is None:
            schedulers = None
        else:
            schedulers = []
        for optim in self.optims:
            if self.args.scheduler_mode == 'cosine':
                scheduler = CosineAnnealingLR(optim, T_max=self.args.T_max)
            elif self.args.scheduler_mode == 'onecycle':
                scheduler = OneCycleLR(optim, max_lr=0.05,
                                    steps_per_epoch=(100000//self.args.batch_size),
                                    epochs=self.args.epochs, div_factor=10,
                                    final_div_factor=50, pct_start=0.3)
            schedulers.append(scheduler)
            
        return schedulers

    def initDls(self):
        trn_dls, val_dls = get_dl_lists(dataset=self.args.dataset, partition=self.args.partition, n_site=self.args.site_number, batch_size=self.args.batch_size, alpha=self.args.alpha)
        return trn_dls, val_dls

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.args.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        trn_dls, val_dls = self.initDls()
        log.debug('initiated dls')

        saving_criterion = 0
        validation_cadence = 5
        for epoch_ndx in range(1, self.args.epochs + 1):

            if epoch_ndx == 1:
                log.info("Epoch {} of {}, training on {} sites, using {} device".format(
                    epoch_ndx,
                    self.args.epochs,
                    len(trn_dls),
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

            trn_metrics = self.doTraining(epoch_ndx, trn_dls)
            self.logMetrics(epoch_ndx, 'trn', trn_metrics)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                val_metrics, accuracy, imgs = self.doValidation(epoch_ndx, val_dls)
                self.logMetrics(epoch_ndx, 'val', val_metrics, imgs)
                saving_criterion = max(accuracy, saving_criterion)

                if self.args.save_model:
                    self.saveModel('layer_personalisation', epoch_ndx, accuracy == saving_criterion)

                log.info('Epoch {} of {}, accuracy/miou {}'.format(epoch_ndx, self.args.epochs, accuracy))
            
            if self.args.scheduler_mode == 'cosine':
                for scheduler in self.schedulers:
                    scheduler.step()
                    # log.debug(self.scheduler.get_last_lr())

            if self.args.site_number > 1:
                self.mergeModels()

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, trn_dls):
        for model in self.models:
            model.train()

        trn_metrics = torch.zeros(2 + 2*self.args.site_number, device=self.device)
        loss = 0
        correct = 0
        total = 0
        for ndx, trn_dl in enumerate(trn_dls):
            local_trn_metrics = torch.zeros(3, len(trn_dl), device=self.device)

            for batch_ndx, batch_tuple in enumerate(trn_dl):
                self.optims[ndx].zero_grad()

                loss, _, _ = self.computeBatchLoss(
                    batch_ndx,
                    batch_tuple,
                    self.models[ndx],
                    local_trn_metrics,
                    'trn')

                loss.backward()
                self.optims[ndx].step()
                if self.args.scheduler_mode == 'onecycle':
                    self.schedulers[ndx].step()

            loss += local_trn_metrics[0].sum()
            correct += local_trn_metrics[1].sum()
            total += local_trn_metrics[2].sum()
            trn_metrics[2*ndx] = local_trn_metrics[0].sum() / local_trn_metrics[2].sum()
            trn_metrics[2*ndx + 1] = local_trn_metrics[1].sum() / local_trn_metrics[2].sum()

        trn_metrics[-2] = loss / total
        trn_metrics[-1] = correct / total

        self.totalTrainingSamples_count += len(trn_dls[0].dataset)

        return trn_metrics.to('cpu')

    def doValidation(self, epoch_ndx, val_dls):
        with torch.no_grad():
            for model in self.models:
                model.eval()
            if epoch_ndx == 1:
                log.warning('E{} Validation starting'.format(epoch_ndx))

            val_metrics = torch.zeros(2 + 2*self.args.site_number, device=self.device)
            loss = 0
            correct = 0
            total = 0
            for ndx, val_dl in enumerate(val_dls):
                local_val_metrics = torch.zeros(3, len(val_dl), device=self.device)

                for batch_ndx, batch_tuple in enumerate(val_dl):
                    _, accuracy, imgs = self.computeBatchLoss(
                        batch_ndx,
                        batch_tuple,
                        self.models[ndx],
                        local_val_metrics,
                        'val'
                    )
                
                loss += local_val_metrics[0].sum()
                correct += local_val_metrics[1].sum()
                total += local_val_metrics[2].sum()
                val_metrics[2*ndx] = local_val_metrics[0].sum() / local_val_metrics[2].sum()
                val_metrics[2*ndx + 1] = local_val_metrics[1].sum() / local_val_metrics[2].sum()

            val_metrics[-2] = loss / total
            val_metrics[-1] = correct / total

        return val_metrics.to('cpu'), correct / total, imgs

    def computeBatchLoss(self, batch_ndx, batch_tup, model, metrics, mode):
        batch, labels = batch_tup
        if self.args.dataset == 'pascalvoc':
            batch = batch.to(device=self.device, non_blocking=True)
            labels = labels.to(device=self.device, non_blocking=True).squeeze(dim=1).to(dtype=torch.long)
        else:
            batch = batch.to(device=self.device, non_blocking=True).permute(0, 3, 1, 2).float()
            labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)

        if mode == 'trn':
            assert self.args.aug_mode in ['classification', 'segmentation']
            batch, labels = aug_image(batch, labels, self.args.aug_mode)

        pred = model(batch)
        pred_label = torch.argmax(pred, dim=1)
        if self.args.aug_mode == 'segmentation':
            weight = torch.ones(21, device=self.device, dtype=torch.float)
            weight[0] *= self.args.background_weight
            if len(pred.shape) != len(batch.shape):
                pred = pred.unsqueeze(dim=0)
        else:
            weight = None
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(pred, labels)

        metrics[0, batch_ndx] = loss.detach()
        metrics[2, batch_ndx] = batch.shape[0]

        if self.args.aug_mode == 'classification':
            correct_mask = pred_label == labels
            correct = torch.sum(correct_mask)
            accuracy = correct / batch.shape[0] * 100

            metrics[1, batch_ndx] = correct
        elif self.args.aug_mode == 'segmentation':
            accuracy = batch_miou(pred_label, labels)

            metrics[1, batch_ndx] = accuracy.sum()

        if mode == 'val' and self.args.aug_mode == 'segmentation':
            img_number = min(5, batch.shape[0])
            val_mean = torch.tensor([0.4561, 0.4353, 0.4013], device=self.device)
            val_std = torch.tensor([0.2657, 0.2625, 0.2771], device=self.device)
            original_img = batch[0:img_number].permute(0, 2, 3, 1)
            original_img = original_img*val_std + val_mean
            original_img = original_img.permute(0, 3, 1, 2)
            predicted_mask = pred_label[0:img_number]
            predicted_mask = torch.stack([predicted_mask, predicted_mask, predicted_mask], dim=1)
            original_mask = labels[0:img_number]
            original_mask = torch.stack([original_mask, original_mask, original_mask], dim=1)
            # imgs = [original_img, predicted_mask, original_mask]
            imgs = torch.cat([original_img, predicted_mask, original_mask], dim=0)
        else:
            imgs = None

        return loss.mean(), accuracy, imgs

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics,
        imgs=None
    ):
        self.initTensorboardWriters()

        writer = getattr(self, mode_str + '_writer')
        if self.args.aug_mode == 'classification':
            metric_name = 'accuracy'
        else:
            metric_name = 'miou'
        for ndx in range(self.args.site_number):
            writer.add_scalar(
                'loss/site {}'.format(ndx),
                scalar_value=metrics[2*ndx],
                global_step=epoch_ndx
            )
            writer.add_scalar(
                '{}/site {}'.format(metric_name,ndx),
                scalar_value=metrics[2*ndx + 1],
                global_step=epoch_ndx
            )
        writer.add_scalar(
            'loss/overall',
            scalar_value=metrics[-2],
            global_step=epoch_ndx
        )
        writer.add_scalar(
            '{}/overall'.format(metric_name),
            scalar_value=metrics[-1],
            global_step=epoch_ndx
        )
        if imgs is not None:
            grid = torchvision.utils.make_grid(imgs, nrow=5)
            writer.add_image(
                'images',
                grid,
                global_step=epoch_ndx,
                dataformats='CHW')
        writer.flush()

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        for ndx, model in enumerate(self.models):
            if isBest:
                file_path = os.path.join(
                    'saved_models',
                    self.args.logdir,
                    '{}_{}_{}.site{}.state'.format(
                        type_str,
                        self.time_str,
                        self.args.comment,
                        ndx
                    )
                )

                os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

                if isinstance(model, torch.nn.DataParallel):
                    model = model.module

                state = {
                    'model_state': model.state_dict(),
                    'model_name': type(model).__name__,
                    'optimizer_state': self.optims[0].state_dict(),
                    'optimizer_name': type(self.optims[0]).__name__,
                    'epoch': epoch_ndx,
                    'totalTrainingSamples_count': self.totalTrainingSamples_count
                }

                torch.save(state, file_path)

                log.debug("Saved model params to {}".format(file_path))

    def mergeModels(self):
        layer_list = get_layer_list(model=self.args.model_name, strategy=self.args.strategy)
        state_dicts = [model.state_dict() for model in self.models]
        param_dict = {layer: torch.zeros(state_dicts[0][layer].shape, device=self.device) for layer in layer_list}

        for layer in layer_list:
            for state_dict in state_dicts:
                param_dict[layer] += state_dict[layer]
            param_dict[layer] /= len(state_dicts)

        for model in self.models:
            model.load_state_dict(param_dict, strict=False)

if __name__ == '__main__':
    LayerPersonalisationTrainingApp().main()
