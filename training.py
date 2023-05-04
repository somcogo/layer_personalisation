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

from models.model import ResNet18Model, ResNet34Model, TinySwin, SmallSwin, LargeSwin, UnetWithResNet34
from utils.logconf import logging
from utils.data_loader import get_cifar10_dl, get_cifar100_dl, get_pascalvoc_dl
from utils.ops import aug_image, batch_miou

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class TinyImageNetTrainingApp:
    def __init__(self, sys_argv=None, epochs=None, batch_size=None, logdir=None, lr=None, comment=None, dataset='cifar10', site_number=5, model_name=None, optimizer_type=None, scheduler_mode=None, label_smoothing=None, T_max=None, pretrained=None, aug_mode=None, save_model=None):
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
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.args.logdir)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        self.scheduler = self.initScheduler()

    def initModel(self):
        if self.args.dataset == 'cifar10':
            num_classes = 10
        elif self.args.dataset == 'cifar100':
            num_classes = 100
        elif self.args.dataset == 'pascalvoc':
            num_classes = 21
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
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        if self.args.optimizer_type == 'adam':
            optim = Adam(params=self.model.parameters(), lr=self.args.lr)
        elif self.args.optimizer_type == 'adamw':
            optim = AdamW(params=self.model.parameters(), lr=self.args.lr, weight_decay=0.05)
        elif self.args.optimizer_type == 'sgd':
            optim = SGD(params=self.model.parameters(), lr=self.args.lr, weight_decay=0.0001, momentum=0.9)
        return optim
    
    def initScheduler(self):
        if self.args.scheduler_mode == 'cosine':
            scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.T_max)
        elif self.args.scheduler_mode == 'onecycle':
            scheduler = OneCycleLR(self.optimizer, max_lr=0.05,
                                   steps_per_epoch=(100000//self.args.batch_size),
                                   epochs=self.args.epochs, div_factor=10,
                                   final_div_factor=50, pct_start=0.3)
        else:
            assert self.args.scheduler_mode is None
            scheduler = None
        return scheduler

    def initDl(self):
        if self.args.dataset == 'cifar10':
            log.debug('using cifar10')
            trn_dl, val_dl = get_cifar10_dl(partition='regular', n_sites=1, batch_size=self.args.batch_size)
        elif self.args.dataset == 'cifar100':
            log.debug('using cifar100')
            trn_dl, val_dl = get_cifar100_dl(partition='regular', n_sites=1, batch_size=self.args.batch_size)
        elif self.args.dataset == 'pascalvoc':
            log.debug('using pascalvoc')
            trn_dl, val_dl = get_pascalvoc_dl(partition='regular', n_sites=1, batch_size=self.args.batch_size)
        return trn_dl, val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.args.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        train_dl, val_dl = self.initDl()

        saving_criterion = 0
        validation_cadence = 5
        for epoch_ndx in range(1, self.args.epochs + 1):

            if epoch_ndx == 1 or epoch_ndx % 10 == 0:
                log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.args.epochs,
                    len(train_dl),
                    len(val_dl),
                    self.args.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

            trnMetrics = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics, correct_ratio = self.doValidation(epoch_ndx, val_dl)
                self.logMetrics(epoch_ndx, 'val', valMetrics)
                saving_criterion = max(correct_ratio, saving_criterion)

                if self.args.save_model:
                    self.saveModel('imagenet', epoch_ndx, correct_ratio == saving_criterion)
            
            if self.args.scheduler_mode == 'cosine':
                self.scheduler.step()
                # log.debug(self.scheduler.get_last_lr())

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics = torch.zeros(2, len(train_dl), device=self.device)

        if epoch_ndx == 1 or epoch_ndx % 10 == 0:
            log.warning('E{} Training ---/{} starting'.format(epoch_ndx, len(train_dl)))

        for batch_ndx, batch_tuple in enumerate(train_dl):
            self.optimizer.zero_grad()

            loss, _ = self.computeBatchLoss(
                batch_ndx,
                batch_tuple,
                trnMetrics,
                'trn')

            loss.backward()
            self.optimizer.step()
            if self.args.scheduler_mode == 'onecycle':
                self.scheduler.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics = torch.zeros(2, len(val_dl), device=self.device)

            log.warning('E{} Validation ---/{} starting'.format(epoch_ndx, len(val_dl)))

            for batch_ndx, batch_tuple in enumerate(val_dl):
                _, accuracy = self.computeBatchLoss(
                    batch_ndx,
                    batch_tuple,
                    valMetrics,
                    'val'
                )

        return valMetrics.to('cpu'), accuracy

    def computeBatchLoss(self, batch_ndx, batch_tup, metrics, mode):
        batch, labels = batch_tup
        batch = batch.to(device=self.device, non_blocking=True)
        labels = labels.to(device=self.device, non_blocking=True).squeeze(dim=1)

        if mode == 'trn':
            assert self.args.aug_mode in ['classification', 'segmentation']
            batch, labels = aug_image(batch, labels, self.args.aug_mode)

        pred = self.model(batch)
        pred_label = torch.argmax(pred, dim=1)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        loss = loss_fn(pred, labels)

        if self.args.aug_mode == 'classification':
            correct_mask = pred_label == labels
            correct = torch.sum(correct_mask)
            accuracy = correct / batch.shape[0] * 100
        elif self.args.aug_mode == 'segmentation':
            accuracy = batch_miou(pred_label, labels).mean()

        metrics[0, batch_ndx] = loss.detach()
        metrics[1, batch_ndx] = accuracy

        return loss.mean(), accuracy

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics,
        img_list=None
    ):
        self.initTensorboardWriters()

        if epoch_ndx == 1 or epoch_ndx % 10 == 0:
            log.info(
                "E{} {}:{} loss".format(
                    epoch_ndx,
                    mode_str,
                    metrics[0].mean()
                )
            )

        writer = getattr(self, mode_str + '_writer')
        if self.args.aug_mode == 'classification':
            metric_name = 'accuracy'
        else:
            metric_name = 'miou'
        writer.add_scalar(
            'loss/overall',
            scalar_value=metrics[0].mean(),
            global_step=self.totalTrainingSamples_count
        )
        writer.add_scalar(
            '{}/overall'.format(metric_name),
            scalar_value=metrics[1].mean(),
            global_step=self.totalTrainingSamples_count
        )
        writer.flush()

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'saved_models',
            self.args.logdir,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.args.comment,
                self.totalTrainingSamples_count
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count
        }

        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'saved_models',
                self.args.logdir,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.args.comment,
                    'best'
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

if __name__ == '__main__':
    TinyImageNetTrainingApp().main()
