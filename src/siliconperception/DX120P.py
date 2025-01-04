# PLACEHOLDER
import numpy as np
import torch
import torch.nn as nn
import torchinfo ; print('torchinfo',torchinfo.__version__)
from collections import OrderedDict
class DX120P(
        nn.Module
        ):
    def __init__(self):
        super().__init__()
        self.layer1  = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.layer2  = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer3  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer4  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer5  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer6  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer7  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer8  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer9  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer10  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer11  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(256), nn.ReLU())
        self.layer12  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(256), nn.ReLU())
        self.layer13  = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(256), nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        return out


decoder = DX120P()
torchinfo.summary(decoder,col_names=["input_size","output_size","num_params"],input_size=(1,512,112,112))
