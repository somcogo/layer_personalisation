import torch.nn as  nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, swin_t, swin_s, Swin_T_Weights, Swin_S_Weights
from torchvision.transforms import Resize
from timm import create_model
import segmentation_models_pytorch as smp

from .resnet_with_embeddings import ResNetWithEmbeddings

class ResNet18Model(nn.Module):
    def __init__(self, num_classes,  pretrained=False):
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(512, num_classes)        

    def forward(self, x):
        out = self.model(x)
        return out
    
class ResNet34Model(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        if pretrained:
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None
        self.model = resnet34(weights=weights)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

class TinySwin(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()

        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None
        self.tiny_swin = swin_t(weights=weights)
        self.tiny_swin.head = nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x):
        out = self.tiny_swin(x)
        return out
    
class SmallSwin(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()

        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None
        self.small_swin = swin_s(weights=weights)
        self.small_swin.head = nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x):
        out = self.small_swin(x)
        return out

class LargeSwin(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()

        self.large_swin = create_model('swin_large_patch4_window12_384',
                                     pretrained=pretrained, drop_path_rate=0.1,
                                     num_classes=num_classes)
        self.resize = Resize(size=(384, 384), antialias=True)

    def forward(self, x):
        x = self.resize(x)
        out = self.large_swin(x)
        return out
    
class UnetWithResNet34(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        if pretrained:
            weights = 'imagenet'
        else:
            weights = None
        self.unet = smp.Unet('resnet34', encoder_depth=5, encoder_weights=weights, in_channels=3, classes=num_classes)

    def forward(self, x):
        out = self.unet(x)
        return out.squeeze(dim=0)