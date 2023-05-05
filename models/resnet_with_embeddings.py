from torch import nn

from edmcode import UNetBlock

class ResNetWithEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = UNetBlock(in_channels=16, out_channels=64, emb_channels=8, down=True)