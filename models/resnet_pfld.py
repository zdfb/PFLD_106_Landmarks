import torch
import torch.nn as nn
from torchvision import models


###### 基于resnet50的PFLD #####



def conv_bn(inp, oup, kernel_size, stride, padding=1, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, kernel_size, stride, padding, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

# 定义PFLD的BackBone部分
class PFLDInference(nn.Module):
    def __init__(self, pretrained = True):
        super(PFLDInference, self).__init__()

        model = models.resnet50(pretrained = pretrained)  # 设置为50
        features1 = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
        features2 = list([model.layer4])
        self.backbone1 = nn.Sequential(*features1) # 特征提取网络, 7, 7, 1024
        self.backbone2 = nn.Sequential(*features2) # 特征提取网络, 4, 4, 2048
        
        # 7, 7, 1024 -> 7, 7, 256
        self.conv_1 = conv_bn(1024, 256, 3, 1)
        self.avg_pool1 = nn.AvgPool2d(7)
        
        # 4, 4, 2048 -> 4, 4, 512
        self.conv_2 = conv_bn(2048, 256, 3, 1)
        self.avg_pool2 = nn.AvgPool2d(4)
        
        # 4, 4, 512 -> 1, 1, 256
        self.conv3 = nn.Conv2d(256, 256, 4, 1, 0)
        self.fc = nn.Linear(256 * 3, 106 * 2)

    def forward(self, x):
        out1 = self.backbone1(x)
        out2 = self.backbone2(out1)

        x = self.conv_1(out1)  # 7, 7, 1024 -> 7, 7, 256
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)


        x = self.conv_2(out2)  # 4, 4, 2048 -> 4, 4, 256
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.conv3(x)  # 4, 4, 256 -> 1, 1, 256
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return out2, landmarks


# 定义PFLD的BackBone部分
class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()

        self.conv1 = conv_bn(2048, 128, 3, 1)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 1)
        self.conv4 = conv_bn(32, 128, 3, 1)

        self.max_pool1 = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x
