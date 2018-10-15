import torch 
import torch.nn as nn
import torch.nn.functional as F

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

    residual block:
    - conv 3x3, stride 1, padding 1
    - batchnorm (disabled for now)
    - ELU
    - conv 3x3, stride 1, padding 1
    - batchnorm
    - ELU
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
        self.pool = nn.AvgPool2d(kernel_size=(self.FRAMES, 1), stride=1)
        self.final = nn.Linear(256, self.CLASSES, bias=True)
    
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
        x = torch.squeeze(x)
        x = F.normalize(x, p=2, dim=1)
        x *= self.ALPHA
        # done
        x = self.final(x)
        return x
