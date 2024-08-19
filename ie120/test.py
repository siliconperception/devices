# pretrain IE120 image encoder at 768x768 RGB resolution
import batch
import ie120
import numpy as np
import torch
import torch.nn as nn
import torchinfo
import argparse
import random
import datetime
import queue
import threading
import subprocess
import pickle

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--topn', help='include top n predictions in accuracy calculation',default=5, type=int)
parser.add_argument('--nbatch', help='batches to test',default=1000, type=int)
parser.add_argument('--model', help='finetune model name',default='finetune.pt')
parser.add_argument('--freeze', help='freeze encoder weights',default=False, action='store_true')
parser.add_argument('--alt', help='encoder model alt type',default='alt1')
parser.add_argument('--encoder', help='encoder model name',default='ie120-050-240')
parser.add_argument('--save', help='output encoder model name',default='finetune.pt')
parser.add_argument('--avg', help='moving average window for lr and grad',default=100, type=int)
parser.add_argument('--factor', help='LR schedule param',default=0.1, type=float)
parser.add_argument('--slow', help='reduce momentum to 0.9 at this batch',default=-1, type=int)
parser.add_argument('--steps', help='LR scheduler steps',action='store', type=int,nargs='*')
parser.add_argument('--sched', help='LR scheduler type',default='linear')
parser.add_argument('--gamma', help='LR schedule param',default=1.0, type=float)
parser.add_argument('--nesterov', help='SGD param',default=False, action='store_true')
parser.add_argument('--momentum', help='SGD param',default=0.0, type=float)
parser.add_argument('--dampening', help='SGD param',default=0.0, type=float)
parser.add_argument('--opt', help='optimizer type',default='sgd')
parser.add_argument('--weight_decay', help='L2 penalty',default=0.0, type=float)
parser.add_argument('--roi', help='input image x/y size',default=768, type=int)
parser.add_argument('--centercrop', help='crop to square',default=False, action='store_true')
parser.add_argument('--step', help='LR scheduler batches per step',default=1000000000, type=int)
parser.add_argument('--end_factor', help='LR linear schedule parameter',default=1./50, type=float)
parser.add_argument('--total_iters', help='LR linear schedule parameter',default=10, type=float)
parser.add_argument('--x1', help='include 1x1 examples',default=False, action='store_true')
parser.add_argument('--x2', help='include 2x2 examples',default=False, action='store_true')
parser.add_argument('--x3', help='include 3x3 examples',default=False, action='store_true')
parser.add_argument('--workers', help='number of threads for batch generation',default=20, type=int)
parser.add_argument('--resize', help='resize scale',default=0.0, type=float)
parser.add_argument('--imagenet', help='imagenet dataset base directory',default='../')
parser.add_argument('--scratch', help='start training from random weights',default=False, action='store_true')
parser.add_argument('--checkpoint', help='save timestamped checkpoint every 100000 batches',default=False, action='store_true')
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--train', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--batch', help='batch size',default=50, type=int)
parser.add_argument('--lr', help='initial learning rate',default=0.0005, type=float)
parser.add_argument('--show', help='display batches',default=False, action='store_true')
parser.add_argument('--seed', help='random seed',default=None, type=int)
parser.add_argument('--debug', help='verbose',default=False, action='store_true')
args = parser.parse_args()
args.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(args.seed)))
if args.log is None:
    args.date = subprocess.check_output(['/usr/bin/date', '+%Y.%m.%d-%H.%M.%S'])
    args.date = args.date.decode("utf-8")
    args.date = args.date.rstrip()
    args.log = 'log/log.{}'.format(args.date)
args.centercrop=True
args.x1=True
args.x2=False
args.x3=False
print(args)

class Model(nn.Module):
    def __init__(self, encoder, alt='alt1'):
        super(Model, self).__init__()
        self.encoder = encoder
        self.alt = alt
        if alt=='alt1':
            self.projection=1000
            self.layer1 = nn.Conv2d(512, self.projection, kernel_size=(3,3), stride=1) # linearly project [3,3,512] features to [1000] classes
            self.layer1b = nn.SELU()
            self.layer2 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.layer2b = nn.SELU()
            self.layer3 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.layer3b = nn.SELU()
            self.layer4 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.layer4b = nn.SELU()
            self.layer5 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.layer5b = nn.SELU()
            self.layerl = nn.Conv2d(self.projection, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes

    def forward(self, x):
        fmap = self.encoder(x)
        y = self.layer1b(self.layer1(fmap))
        y = self.layer2b(self.layer2(y))
        y = self.layer3b(self.layer3(y))
        y = self.layer4b(self.layer4(y))
        y = self.layer5b(self.layer5(y))
        y = self.layerl(y)
        return y

model = torch.load('{}'.format(args.model))
model.eval()
print('finetune model loaded')

dataset = batch.Batch(args)

with open(args.log, 'a') as f:
    print(args,file=f)
    print(torchinfo.summary(model, input_size=(1, 3, args.roi, args.roi)),file=f)
if args.debug:
    torchinfo.summary(model, input_size=(1, 3, args.roi, args.roi))
device = torch.device('cuda')
model = model.to(device)

acc=[]
for i in range(args.nbatch):
    (d,l) = dataset.generate_batch(args)
    d=torch.utils.data.default_convert(d).to(device)
    logits = model(d)
    prob = torch.nn.functional.softmax(logits,dim=-3)
    prob = torch.reshape(prob,[-1,1000])
    #l = np.sum(l,axis=(-1,-2))
    l = l[:,:,0]
    
    prob = prob.cpu().detach().numpy()
    acc0=[]
    for j in range(args.batch):
        match=False
        for n in range(args.topn):
            pred = np.argsort(prob[j])[-1*(1+n)]
            gt = np.argsort(l[j])[-1]
            #print('j',j,'n',n,'pred',pred,'gt',gt)
            if pred==gt:
                match=True
            #if args.debug:
            #    print('i {:4d} n {:4d} prob {:.6f} pred {:6d} text {:40} gt {:6d} text {:40}'.format(
            #        i,n,prob[pred],pred+1,labeltext[pred+1][0:40],gt,labeltext[gt][0:40]))
        acc.append(match)
        acc0.append(match)
    #print('i',i,'acc',np.mean(acc),'acc0',np.mean(acc0),'n',len(acc),len(acc0))
    print('n {:12d} acc {:12.8f} std {:12.8f}'.format(len(acc),np.mean(acc),np.std(acc)))
