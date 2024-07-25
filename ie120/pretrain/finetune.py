# load encoder model, add task to predict imagenet classes for four quadrants
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import IE120
import numpy as np
import torch
import torch.nn as nn
import torchinfo
import argparse
import collections
import cv2
import scipy
import random
import datetime
import queue
import threading
import Imagenet
import subprocess

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--centercrop', help='crop to square',default=False, action='store_true')
parser.add_argument('--step', help='LR scheduler batches per step',default=10000, type=int)
parser.add_argument('--end_factor', help='LR linear schedule parameter',default=1./50, type=float)
parser.add_argument('--total_iters', help='LR linear schedule parameter',default=10, type=float)
parser.add_argument('--workers', help='number of threads for batch generation',default=20, type=int)
parser.add_argument('--freeze', help='freeze encoder weights',default=False, action='store_true')
parser.add_argument('--imagenet', help='imagenet dataset base directory',default='../../')
parser.add_argument('--scratch', help='start training from random weights',default=False, action='store_true')
parser.add_argument('--checkpoint', help='save timestamped checkpoint every 1000 batches',default=False, action='store_true')
parser.add_argument('--save', help='save temporary checkpoint every 1000 batches',default=False, action='store_true')
parser.add_argument('--alt', help='alternative training tasks',default='alt2')
parser.add_argument('--resize', help='resize images',default=False, action='store_true')
parser.add_argument('--flip', help='flip images',default=False, action='store_true')
parser.add_argument('--scale', help='resize scale',default=0.0, type=float)
parser.add_argument('--encoder', help='encoder model name',default='ie120-050-240')
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--train', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--batch', help='batch size',default=10, type=int)
parser.add_argument('--lr', help='learning rate',default=0.001, type=float)
parser.add_argument('--show', help='display batches',default=False, action='store_true')
parser.add_argument('--seed', help='random seed',default=None, type=int)
parser.add_argument('--debug', help='verbose',default=False, action='store_true')
args = parser.parse_args()
args.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(args.seed)))
if args.log is None:
    args.date = subprocess.check_output(['/usr/bin/date', '+%Y.%m.%d-%H.%M.%S'])
    args.date = args.date.decode("utf-8")
    args.date = args.date.rstrip()
    args.log = 'checkpoint/log.{}'.format(args.date)
print(args)
with open(args.log, 'a') as f:
    print(args,file=f)

synlabel={}
labeltext={}
meta = scipy.io.loadmat('{}/imagenet/devkit-1.0/data/meta.mat'.format(args.imagenet))
synsets = meta['synsets']
for i,s in enumerate(synsets):
    synlabel[s[0][1][0]] = s[0][0][0][0]
    labeltext[s[0][0][0][0]] = s[0][2][0]
    #for j,r in enumerate(s[0]):
    #    print('i',i,'j',j,'r',r)
#print('synlabel',synlabel)
#print('labeltext',labeltext)
print('imagenet metadata loaded')

with open('{}/imagenet/train/flist'.format(args.imagenet), 'r') as f:
    flist = f.readlines()
#flist = random.choices(flist,k=10) # DEBUG
print('imagenet flist loaded',len(flist))

#if args.scratch:
#    encoder = IE120.IE120(args.encoder)
#    print('image encoder model initialized')
#else:
encoder = torch.load('{}'.format(args.encoder))
encoder.eval()
print('image encoder model loaded')

if args.freeze:
    for param in encoder.parameters():
        param.requires_grad = False
    print('image encoder model frozen')

#class Imagenet(nn.Module):
#    def __init__(self, encoder):
#        super(Imagenet, self).__init__()
#        self.encoder = encoder
#        #self.task = nn.Sequential(nn.Conv2d(512, 1000, kernel_size=1, stride=1)) # linearly project [2,2,512] features to [2,2,1000] classes
#        #self.task = nn.Sequential(nn.Flatten(), nn.Linear(encoder.fm_width*encoder.fm_height*512, encoder.lvec))
#        #self.layer18 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1), nn.BatchNorm2d(512), nn.ReLU())
#        self.layer1 = nn.Flatten()
#        self.layer2a = nn.Linear(2048,1000)
#        self.layer2b = nn.Linear(2048,1000)
#        self.layer2c = nn.Linear(2048,1000)
#        self.layer2d = nn.Linear(2048,1000)
#
#    def forward(self, x):
#        fmap = self.encoder(x)
#        f = self.layer1(fmap)
#        y0 = self.layer2a(f)
#        y1 = self.layer2b(f)
#        y2 = self.layer2c(f)
#        y3 = self.layer2d(f)
#        y = torch.cat([y0,y1,y2,y3])
#        y = torch.reshape(y,[-1,1000,2,2])
#        return y

def generate_batch(args,flist,synlabel,labeltext):
    d = np.zeros([args.batch,700,700,3]).astype(np.uint8)
    l = np.zeros([args.batch]).astype(np.int64)
    l0 = np.zeros([args.batch,2,2]).astype(np.int64)
    l1 = np.zeros([args.batch,2,2]).astype(np.int64)
    l2 = np.zeros([args.batch,2,2]).astype(np.int64)
    l3 = np.zeros([args.batch,2,2]).astype(np.int64)
     
    #random.shuffle(flist)
    #for i in range(0,len(flist),stride):
    for i in range(args.batch):
        if args.alt=='alt2':
            s = random.choices(flist,k=1)
        else:
            s = random.choices(flist,k=4)
            if args.scale>0:
                o = np.random.randint(350-args.scale*350,350+args.scale*350,size=2)
            else:
                o = (350,350)
        for j,fn in enumerate(s):
            while True:
                img = cv2.imread('{}/imagenet/train/{}'.format(args.imagenet,fn.rstrip()))
                if img is None:
                    fn = random.choice(flist)
                else:
                    break

            if args.centercrop:
                side = min(img.shape[0],img.shape[1])
                img = img[img.shape[0]//2-side//2:img.shape[0]//2+side//2,img.shape[1]//2-side//2:img.shape[1]//2+side//2]

            if args.alt=='alt2':
                l[i] = synlabel[fn[0:9]]-1
                d[i] = cv2.resize(img,dsize=(700,700),interpolation=cv2.INTER_LINEAR)
            else:
                l0[i,j//2,j%2] = synlabel[fn[0:9]]-1

                if args.flip:
                    f = np.random.randint(4)
                    l3[i,j//2,j%2] = f
                    if f==1:
                        img = cv2.flip(img,1) # horizontal flip
                    if f==2:
                        img = cv2.flip(img,0) # vertical flip
                    if f==3:
                        img = cv2.flip(img,1)
                        img = cv2.flip(img,0)

                if args.resize:
                    if j==0:
                        ulx = 0
                        uly = 0
                        lrx = o[0]
                        lry = o[1]
                        l1[i,j//2,j%2] = o[0]
                        l2[i,j//2,j%2] = o[1]
                    if j==1:
                        ulx = o[0]
                        uly = 0
                        lrx = 700
                        lry = o[1]
                        l1[i,j//2,j%2] = 700-o[0]
                        l2[i,j//2,j%2] = o[1]
                    if j==2:
                        ulx = 0
                        uly = o[1]
                        lrx = o[0]
                        lry = 700
                        l1[i,j//2,j%2] = o[0]
                        l2[i,j//2,j%2] = 700-o[1]
                    if j==3:
                        ulx = o[0]
                        uly = o[1]
                        lrx = 700
                        lry = 700
                        l1[i,j//2,j%2] = 700-o[0]
                        l2[i,j//2,j%2] = 700-o[1]
                    img = cv2.resize(img,dsize=(lrx-ulx,lry-uly),interpolation=cv2.INTER_LINEAR)
                    d[i,uly:lry,ulx:lrx] = img
                else:
                    img = cv2.resize(img,dsize=(350,350),interpolation=cv2.INTER_LINEAR)
                    x0 = 350*(j%2)
                    y0 = 350*(j//2)
                    d[i,y0:y0+350,x0:x0+350] = img


    #cv2.imshow('imagenet', cv2.resize(d[0],dsize=(dw,dh),interpolation=cv2.INTER_LINEAR)) 
    #cv2.imshow('imagenet', d[0])
    #print('l',l[0])
    #cv2.waitKey(1)
    d = d.astype(np.float32)/255.
    d = np.rollaxis(d,-1,1)
    return d,l,l0,l1,l2,l3
    #if args.alt=='alt1':
    #    return d,l0
    #if args.alt=='alt2':
    #    return d,[l0-,l0,l1,l2]
    #if args.alt=='alt3':
    #    return d,[l,l0,l1,l2]
    #else:
    #    return d,l

class Worker(threading.Thread):
    def __init__(self, q, args, flist, synlabel, labeltext):
        self.q = q
        self.args = args
        self.flist = flist
        self.synlabel = synlabel
        self.labeltext = labeltext
        self.running = True
        super().__init__()
    def run(self):
        while self.running:
            (d,l,l0,l1,l2,l3) = generate_batch(self.args,self.flist,self.synlabel,self.labeltext)
            q.put((d,l,l0,l1,l2,l3))

model = Imagenet.Imagenet(encoder,args.alt)
if args.debug:
    torchinfo.summary(model, input_size=(1, 3, 700, 700))
device = torch.device('cuda')
model = model.to(device)

# Loss and optimizer
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
#torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=args.end_factor, total_iters=args.total_iters)
#optimizer = torch.optim.Adam(model.parameters())
#for g in optimizer.param_groups:
#    print(type(g),g.keys(),g['lr'])

q = queue.Queue(maxsize=args.workers)
for _ in range(args.workers):
    w = Worker(q,args,flist,synlabel,labeltext)
    w.daemon = True
    w.start()

#for i in range(args.train):
i=0
larr=[]
garr=[]
while i<args.train:
    (x0,y,y0,y1,y2,y3)=q.get()
    y = np.expand_dims(y,axis=-1)
    y = np.expand_dims(y,axis=-1)

    x=torch.utils.data.default_convert(x0)
    x = x.to(device)
    y=torch.utils.data.default_convert(y)
    y = y.to(device)
    y0=torch.utils.data.default_convert(y0)
    y0 = y0.to(device)
    y1=torch.utils.data.default_convert(y1)
    y1 = y1.to(device)
    y2=torch.utils.data.default_convert(y2)
    y2 = y2.to(device)
    y3=torch.utils.data.default_convert(y3)
    y3 = y3.to(device)
        
    outputs = model(x)
    if args.alt=='alt3':
        loss = criterion(outputs[0], y0)+criterion(outputs[1], y1)+criterion(outputs[2], y2)+criterion(outputs[3], y3)
    elif args.alt=='alt2':
        loss = criterion(outputs, y)
    else:
        loss = criterion(outputs, y0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i%args.step)==0 and i>0:
        scheduler.step()

    # compute gradient
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    #print ('batch [{}/{}] loss {:.4f} grad {:.4f}'.format(i+1, args.train, loss.item(),total_norm))
    larr.append(loss.item())
    garr.append(total_norm)
    lr = optimizer.param_groups[0]['lr']
    #s = 'BATCH {:12d} wall {} lr {:12.10f} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f}'.format(i+1,datetime.datetime.now(),lr,loss.item(),np.mean(larr[-100:]),total_norm,np.mean(garr[-100:]))
    s = 'BATCH {:12d} wall {} lr {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f}'.format(
        i,datetime.datetime.now(),lr,args.batch,loss.item(),np.mean(larr[-100:]),total_norm,np.mean(garr[-100:]))
    print(s)
    with open(args.log, 'a') as f:
        print(s,file=f)
    
    if (i%1000)==0:
        if args.save:
            #torch.save(encoder, '{}'.format(args.encoder))
            torch.save(model, 'finetune.pt')
        #if args.checkpoint:
        #    torch.save(encoder, 'checkpoint/{}.{}.pt'.format(args.encoder,args.date))
        #    torch.save(model, 'checkpoint/pretrain.{}.pt'.format(args.date))

    if (i%1)==0:
        img = x0[0]*255
        img = img.astype(np.uint8)
        img = np.swapaxes(img,0,-1)
        img = np.swapaxes(img,0,1)
        #print('img',img.shape,img.dtype)
        cv2.imshow('imagenet', img)
        cv2.waitKey(1)

    i+=1
