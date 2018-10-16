import torch 
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """
    Simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        a, b = list(x.size())[:2]
        return x.view(a, b)

class ResidualBlock(nn.Module):
    """Residual block
    - conv 3x3, stride 1, padding 1
    - batchnorm (disabled for now)
    - ELU
    - conv 3x3, stride 1, padding 1
    - batchnorm
    - ELU
    """
    def __init__(self, channels):
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.elu2 = nn.ELU()
    
    def forward(self, x):
        save = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = save + x 
        x = self.elu2(x)
        return x

class Resnet(nn.Module):
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
    """
    def __init__(self, classes, alpha=16, frames=14000):
        super(Resnet, self).__init__()
        # constants
        self.CLASSES = classes
        self.ALPHA = alpha
        self.FRAMES = frames
        # first layer
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2, padding=0, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.elu1 = nn.ELU()
        # residual
        self.res1 = ResidualBlock(32)
        # second layers
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=0, bias=False)
        self.elu2 = nn.ELU()
        # residual
        self.res2 = ResidualBlock(64)
        # third layer
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=0, bias=False)
        self.elu3 = nn.ELU()
        # residual
        self.res3 = ResidualBlock(128)
        # fourth layer
        self.conv4 = nn.Conv2d(128, 256, 5, stride=2, padding=0, bias=False)
        self.elu4 = nn.ELU()
        # residual
        self.res4 = ResidualBlock(256)
        # final pooling and length normalizing
        self.pool = nn.AvgPool2d(kernel_size=(self.FRAMES, 1), stride=1)
        self.flatten = Flatten()
        self.final = nn.Linear(256, self.CLASSES, bias=True)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        # first layer
        print('Dimensions before 1st layer:', x.size())
        x = self.conv1(x)
        x = self.elu1(x)
        # residual
        print('Dimensions before 1st res:', x.size())
        x = self.res1(x)
        # second layer
        print('Dimensions before 2nd layer:', x.size())
        x = self.conv2(x)
        x = self.elu2(x)
        # residual
        print('Dimensions before 2nd res:', x.size())
        x = self.res2(x)
        # third layer
        print('Dimensions before 3rd layer:', x.size())
        x = self.conv3(x)
        x = self.elu3(x)
        # residual
        print('Dimensions before 3rd res:', x.size())
        x = self.res3(x)
        # fourth layer
        print('Dimensions before 4th layer:', x.size())
        x = self.conv4(x)
        x = self.elu4(x)
        # residual
        print('Dimensions before 4th res:', x.size())
        x = self.res4(x)
        # final pooling and normalizing
        print('Dimensions before pooling:', x.size())
        x = self.pool(x)
        x = torch.squeeze(x)
        x = F.normalize(x, p=2, dim=1)
        x *= self.ALPHA
        # flatten
        print('Dimenstion before flatten:', x.size())
        x = self.flatten(x)
        # fully connected layer
        print('Dimenstion before linear:', x.size())
        x = self.final(x)
        print('Final dimensionality:', x.size())
        return x
