import torch
from torch import nn

from .edmcode import UNetBlock

class ResNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes, in_channels=3, embed_dim=1, layers=[3, 4, 6, 3]):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(layers[0], in_channels=64, out_channels=64, embed_dim=embed_dim)
        self.layer1 = self._make_layer(layers[1], in_channels=64, out_channels=128, embed_dim=embed_dim)
        self.layer2 = self._make_layer(layers[2], in_channels=128, out_channels=256, embed_dim=embed_dim)
        self.layer3 = self._make_layer(layers[3], in_channels=256, out_channels=512, embed_dim=embed_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, depth, in_channels, out_channels, embed_dim):
        blocks = nn.ModuleList()
        blocks.append(UNetBlock(in_channels=in_channels, out_channels=out_channels, emb_channels=embed_dim))

        for _ in range(1, depth):
            blocks.append(UNetBlock(in_channels=out_channels, out_channels=out_channels, emb_channels=embed_dim))

        return blocks

    def forward(self, x, latent_vector):

        x = self.pool(self.relu(self.norm1(self.conv1(x))))
        latent_vector = latent_vector.repeat(x.shape[0]).view(x.shape[0], -1)
        for block in self.layer0:
            x = block(x, latent_vector)
        for block in self.layer1:
            x = block(x, latent_vector)
        for block in self.layer2:
            x = block(x, latent_vector)
        for block in self.layer3:
            x = block(x, latent_vector)

        out = self.fc(self.avgpool(x).squeeze())
        return out