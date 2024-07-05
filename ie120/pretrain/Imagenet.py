import torch
import torch.nn as nn
class Imagenet(nn.Module):
    def __init__(self, encoder, alt='alt1'):
        super(Imagenet, self).__init__()
        self.encoder = encoder
        self.alt = alt
        if alt=='alt1':
            self.task = nn.Conv2d(512, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
        if alt=='alt2':
            self.layer1 = nn.Flatten()
            self.layer2 = nn.Linear(2048,1000)
            self.layer2a = nn.BatchNorm1d(1000)
            self.layer2b = nn.ReLU()
            self.layer3 = nn.Linear(1000,1000)
            self.layer3a = nn.BatchNorm1d(1000)
            self.layer3b = nn.ReLU()
            self.layer4 = nn.Linear(1000,1000)

            #self.layer3 = nn.Linear(1000,1000)
            #self.layer3b = nn.ReLU()
            #self.layer4 = nn.Linear(1000,1000)
            #self.layer4b = nn.ReLU()
            #self.layer5 = nn.Linear(1000,1000)
            #self.layer5b = nn.SELU()
            #self.layer6 = nn.Linear(1000,1000)
            #self.layer6b = nn.SELU()
            #self.layer7 = nn.Linear(1000,1000)
            #self.layer5a = nn.BatchNorm1d(1000)
            #self.layer5b = nn.ReLU()
            #self.layer6 = nn.Linear(1000,1000)
        if alt=='alt3':
            self.classify = nn.Conv2d(512, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.xscale = nn.Conv2d(512, 700, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.yscale = nn.Conv2d(512, 700, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.xyflip = nn.Conv2d(512, 4, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes

    def forward(self, x):
        fmap = self.encoder(x)
        if self.alt=='alt1':
            y = self.task(fmap)
            return y
        if self.alt=='alt2':
            f = self.layer1(fmap)
            f = self.layer2b(self.layer2a(self.layer2(f)))
            f = self.layer3b(self.layer3a(self.layer3(f)))
            f = self.layer4(f)
            #f = self.layer2(f)
            #f = self.layer3b(self.layer3(f))
            #f = self.layer4(f)
            #f = self.layer3b(self.layer3(f))
            #f = self.layer4b(self.layer4(f))
            #f = self.layer5b(self.layer5(f))
            #f = self.layer6b(self.layer6(f))
            #f = self.layer7(f)
            return f
        if self.alt=='alt3':
            return self.classify(fmap), self.xscale(fmap), self.yscale(fmap), self.xyflip(fmap)


