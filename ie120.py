# This file produces reference PyTorch models for the IE120 image encoder
# It also trains the encoder to perform a synthetic object detection task as a unit test

import numpy as np
import torch
import torch.nn as nn
import torchinfo
import argparse
import collections
import cv2

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', help='model name',default='ie120-050-240')
parser.add_argument('--stretch', help='unit test receptive field random scale factor',default=0.9, type=float)
parser.add_argument('--nbatch', help='unit test training batches',default=10000, type=int)
parser.add_argument('--batch', help='batch size',default=3, type=int)
parser.add_argument('--lr', help='learning rate',default=0.0001, type=float)
parser.add_argument('--show', help='display batches',default=False, action='store_true')
parser.add_argument('--seed', help='random seed',default=None, type=int)
args = parser.parse_args()
args.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(args.seed)))
print(args)

class IE120(nn.Module):
    def __init__(self, args):
        super(IE120, self).__init__()
        if args.model=='ie120-050-240':
            self.width = 700
            self.height = 700
            self.fm_width = 2
            self.fm_height = 2
            self.spacex = 100
            self.spacey = 100
            self.lvec = 49

        if args.model=='ie120-200-060':
            self.width = 1920
            self.height = 1080
            self.fm_width = 21
            self.fm_height = 8
            self.spacex = 192
            self.spacey = 108
            self.lvec = 100

        if args.model=='ie120-500-024':
            self.width = 2448
            self.height = 2048
            self.fm_width = 29
            self.fm_height = 23
            self.spacex = 306
            self.spacey = 256
            self.lvec = 64

        self.layer1  = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2), nn.BatchNorm2d(16), nn.ReLU())
        self.layer2  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1), nn.BatchNorm2d(16), nn.ReLU())
        self.layer3  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1), nn.BatchNorm2d(16), nn.ReLU())
        self.layer4  = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.BatchNorm2d(32), nn.ReLU())
        self.layer5  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer6  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer7  = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU())
        self.layer8  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer9  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer10 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2), nn.BatchNorm2d(128), nn.ReLU())
        self.layer11 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer12 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer13 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.BatchNorm2d(256), nn.ReLU())
        self.layer14 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer15 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer16 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2), nn.BatchNorm2d(512), nn.ReLU())
        self.layer17 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1), nn.BatchNorm2d(512), nn.ReLU())
        self.layer18 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1), nn.BatchNorm2d(512), nn.ReLU())
        self.task = nn.Sequential(nn.Flatten(), nn.Linear(self.fm_width*self.fm_height*512, self.lvec))

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
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        out = self.task(out)
        return out

    def fuse_conv_and_bn(self,conv,bn):
        fusedconv = torch.nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=True)
        w_eff = conv.weight.clone()
        b_eff = conv.bias.clone()
        with torch.no_grad():
            for i in range(conv.out_channels):
                w_eff[i,:,:,:] = torch.div(torch.mul(bn.weight[i],w_eff[i,:,:,:]),torch.sqrt(bn.running_var[i]+bn.eps))
                b_eff[i] = torch.add(torch.div(torch.mul(bn.weight[i],torch.sub(b_eff[i],bn.running_mean[i])),torch.sqrt(bn.running_var[i]+bn.eps)),bn.bias[i])
        fusedconv.weight.data = w_eff.data
        fusedconv.bias.data = b_eff.data
        return fusedconv

    def save(self, args):
        # merge BN with previous Conv2D
        enc = nn.Sequential(collections.OrderedDict([
          ('conv1', self.fuse_conv_and_bn(self.layer1[0],self.layer1[1])), ('relu1', nn.ReLU()),
          ('conv2', self.fuse_conv_and_bn(self.layer2[0],self.layer2[1])), ('relu2', nn.ReLU()),
          ('conv3', self.fuse_conv_and_bn(self.layer3[0],self.layer3[1])), ('relu3', nn.ReLU()),
          ('conv4', self.fuse_conv_and_bn(self.layer4[0],self.layer4[1])), ('relu4', nn.ReLU()),
          ('conv5', self.fuse_conv_and_bn(self.layer5[0],self.layer5[1])), ('relu5', nn.ReLU()),
          ('conv6', self.fuse_conv_and_bn(self.layer6[0],self.layer6[1])), ('relu6', nn.ReLU()),
          ('conv7', self.fuse_conv_and_bn(self.layer7[0],self.layer7[1])), ('relu7', nn.ReLU()),
          ('conv8', self.fuse_conv_and_bn(self.layer8[0],self.layer8[1])), ('relu8', nn.ReLU()),
          ('conv9', self.fuse_conv_and_bn(self.layer9[0],self.layer9[1])), ('relu9', nn.ReLU()),
          ('conv10', self.fuse_conv_and_bn(self.layer10[0],self.layer10[1])), ('relu10', nn.ReLU()),
          ('conv11', self.fuse_conv_and_bn(self.layer11[0],self.layer11[1])), ('relu11', nn.ReLU()),
          ('conv12', self.fuse_conv_and_bn(self.layer12[0],self.layer12[1])), ('relu12', nn.ReLU()),
          ('conv13', self.fuse_conv_and_bn(self.layer13[0],self.layer13[1])), ('relu13', nn.ReLU()),
          ('conv14', self.fuse_conv_and_bn(self.layer14[0],self.layer14[1])), ('relu14', nn.ReLU()),
          ('conv15', self.fuse_conv_and_bn(self.layer15[0],self.layer15[1])), ('relu15', nn.ReLU()),
          ('conv16', self.fuse_conv_and_bn(self.layer16[0],self.layer16[1])), ('relu16', nn.ReLU()),
          ('conv17', self.fuse_conv_and_bn(self.layer17[0],self.layer17[1])), ('relu17', nn.ReLU()),
          ('conv18', self.fuse_conv_and_bn(self.layer18[0],self.layer18[1])), ('relu18', nn.ReLU()),
        ]))
        task = nn.Sequential(collections.OrderedDict([('task', self.task)]))
        # save model to file
        torch.save(enc, '{}.pt'.format(args.model))
        torch.save(task, 'task.{}.pt'.format(args.model))

# Train the model
def generate_batch(args,width,height,lvec,spacex,spacey):
    rs = args.rs
    d = np.zeros([args.batch,height,width,3]).astype(np.uint8)
    l = np.zeros([args.batch,lvec]).astype(np.float32)

    # randomly render colored gaussian balls on grid
    grid=[(x+spacex//2,y+spacey//2) for x in range(0,width,spacex) for y in range(0,height,spacey)]
    for i in range(args.batch):
        for j in range(lvec):
            if rs.choice([True,False]):
                l[i,j]=1
                xy=rs.normal(loc=grid[j],scale=10,size=[2000,2])
                color=tuple(rs.choice([128,255]) for _ in range(3))
                for (x,y) in ((x,y) for (x,y) in xy if x>=0 and x<width and y>=0 and y<height):
                    d[i,int(y),int(x)]=color

    # randomly stretch and center
    for i in range(args.batch):
        fx=(1-args.stretch)+rs.uniform()*args.stretch
        fy=(1-args.stretch)+rs.uniform()*args.stretch
        img=cv2.resize(d[i],dsize=(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
        d[i].fill(0)
        wy=img.shape[0]
        wx=img.shape[1]
        d[i,height//2-wy//2:height//2-wy//2+wy,width//2-wx//2:width//2-wx//2+wx,:]=img

    # optionally display first sample in batch
    if args.show:
        dw=width
        dh=height
        while dw>1000:
            dw = dw//2
            dh = dh//2
        cv2.imshow('ie120', cv2.resize(d[0],dsize=(dw,dh),interpolation=cv2.INTER_LINEAR)) 
        cv2.waitKey(1)
    d = d.astype(np.float32)/255.
    d = np.rollaxis(d,-1,1)
    return (d,l)

# Device configuration
model = IE120(args)
torchinfo.summary(model, input_size=(1, 3, model.height, model.width))
device = torch.device('cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for i in range(args.nbatch):
    (x,y)=generate_batch(args,model.width,model.height,model.lvec,model.spacex,model.spacey)
    x=torch.utils.data.default_convert(x)
    y=torch.utils.data.default_convert(y)
    x = x.to(device)
    y = y.to(device)
        
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    print ('batch [{}/{}] loss {:.4f} grad {:.4f}'.format(i+1, args.nbatch, loss.item(),total_norm))
    if (i%100)==0:
        model.save(args)
