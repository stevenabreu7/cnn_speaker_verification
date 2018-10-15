import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """
    Implementing a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        a, b = list(x.size())[:2]
        return x.view(a, b)

def make_network():
    """Network
    All operations are 2D (frames x features), one channel.
    - Conv 5x5, 32 output channels, stride 2, padding 0
    - residual block
    - Conv 5x5, 64 output channels, stride 2, padding 0
    - residual block
    - Conv 5x5, 128 output channels, stride 2, padding 0
    - residual block
    - Conv 5x5, 256 output channels, stride 2, padding 0
    - residual block
    - Avg Pool, 6x6
    - Flatten
    residual block:
    - conv 3x3, stride 1, padding 1
    - batchnorm (disabled for now)
    - ELU
    - conv 3x3, stride 1, padding 1
    - batchnorm
    - ELU
    """
    def residual_block(channels):
        return [
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ELU(), 
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ELU()
        ]
    
    layers = [
        nn.Conv2d(1, 32, 5, stride=2, padding=0, bias=False),
        nn.ELU(),
        *residual_block(32),
        nn.Conv2d(32, 64, 5, stride=2, padding=0, bias=False),
        nn.ELU(),
        *residual_block(64),
        nn.Conv2d(64, 128, 5, stride=2, padding=0, bias=False),
        nn.ELU(),
        *residual_block(128),
        nn.Conv2d(128, 256, 5, stride=2, padding=0, bias=False),
        nn.ELU(),
        *residual_block(256),
        nn.AvgPool2d(6),
        Flatten()
    ]
    return nn.Sequential(*layers)

class MyNetwork(nn.Module):
    """Network
    All operations are 2D (frames x features), one channel.
    - Conv 5x5, 32 output channels, stride 2, padding 0
    - residual block
    - Conv 5x5, 64 output channels, stride 2, padding 0
    - residual block
    - Conv 5x5, 128 output channels, stride 2, padding 0
    - residual block
    - Conv 5x5, 256 output channels, stride 2, padding 0
    - residual block
    - Avg Pool, 6x6, across time
    - Length normalization (by L2 norm), then scaling

    residual block:
    - conv 3x3, stride 1, padding 1
    - batchnorm (disabled for now)
    - ELU
    - conv 3x3, stride 1, padding 1
    - batchnorm
    - ELU
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        # constants
        self.ALPHA = 16
        # first layer
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2, padding=0, bias=False)
        self.elu1 = nn.ELU()
        # residual
        self.res_conv1_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.res_bn1_1 = nn.BatchNorm2d(32)
        self.res_elu1_1 = nn.ELU()
        self.res_conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.res_bn1_2 = nn.BatchNorm2d(32)
        self.res_elu1_2 = nn.ELU()
        # second layers
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=0, bias=False)
        self.elu2 = nn.ELU()
        # residual
        self.res_conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.res_bn2_1 = nn.BatchNorm2d(64)
        self.res_elu2_1 = nn.ELU()
        self.res_conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.res_bn2_2 = nn.BatchNorm2d(64)
        self.res_elu2_2 = nn.ELU()
        # third layer
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=0, bias=False)
        self.elu3 = nn.ELU()
        # residual
        self.res_conv3_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.res_bn3_1 = nn.BatchNorm2d(128)
        self.res_elu3_1 = nn.ELU()
        self.res_conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.res_bn3_2 = nn.BatchNorm2d(128)
        self.res_elu3_2 = nn.ELU()
        # fourth layer
        self.conv4 = nn.Conv2d(128, 256, 5, stride=2, padding=0, bias=False)
        self.elu4 = nn.ELU()
        # residual
        self.res_conv4_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.res_bn4_1 = nn.BatchNorm2d(256)
        self.res_elu4_1 = nn.ELU()
        self.res_conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.res_bn4_2 = nn.BatchNorm2d(256)
        self.res_elu4_2 = nn.ELU()
        # final pooling and length normalizing
        self.pool = nn.AvgPool2d(6)
        # self.len_norm = 
    
    def forward(self, x):
        # first layer
        x = self.conv1(x)
        x = self.elu1(x)
        # residual
        save = x
        x = self.res_conv1_1(x)
        x = self.res_bn1_1(x)
        x = self.res_elu1_1(x)
        x = self.res_conv1_2(x)
        x = self.res_bn1_2(x)
        x = save + x
        x = self.res_elu1_2(x)
        # second layer
        x = self.conv2(x)
        x = self.elu2(x)
        # residual
        save = x
        x = self.res_conv2_1(x)
        x = self.res_bn2_1(x)
        x = self.res_elu2_1(x)
        x = self.res_conv2_2(x)
        x = self.res_bn2_2(x)
        x = save + x
        x = self.res_elu2_2(x)
        # third layer
        x = self.conv3(x)
        x = self.elu3(x)
        # residual
        save = x
        x = self.res_conv3_1(x)
        x = self.res_bn3_1(x)
        x = self.res_elu3_1(x)
        x = self.res_conv3_2(x)
        x = self.res_bn3_2(x)
        x = save + x
        x = self.res_elu3_2(x)
        # fourth layer
        x = self.conv4(x)
        x = self.elu4(x)
        # residual
        save = x
        x = self.res_conv4_1(x)
        x = self.res_bn4_1(x)
        x = self.res_elu4_1(x)
        x = self.res_conv4_2(x)
        x = self.res_bn4_2(x)
        x = save + x
        x = self.res_elu4_2(x)
        # final pooling and normalizing
        x = self.pool(x)
        x = F.normalize(x, p=2, dim=1)
        x *= self.ALPHA
        # done
        return x
