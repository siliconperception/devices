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
import Imagenet

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--finetune', help='test single image finetuned imagenet classifier',default=False, action='store_true')
parser.add_argument('--imagenet', help='imagenet dataset base directory',default='../../')
parser.add_argument('--alt', help='pretrain arch',default='alt3')
parser.add_argument('--topn', help='include top n predictions in accuracy calculation',default=5, type=int)
parser.add_argument('--model', help='pytorch model filename',default='pretrain.pt')
parser.add_argument('--single', help='test single image',default=False, action='store_true')
parser.add_argument('--quad', help='test 4-quadrant image',default=False, action='store_true')
#parser.add_argument('--encoder', help='encoder model name',default='ie120-050-240')
#parser.add_argument('--log', help='log file name',default='log')
#parser.add_argument('--train', help='total training batches',default=1000000000000, type=int)
#parser.add_argument('--batch', help='batch size',default=3, type=int)
#parser.add_argument('--lr', help='learning rate',default=0.0001, type=float)
#parser.add_argument('--show', help='display batches',default=False, action='store_true')
parser.add_argument('--seed', help='random seed',default=None, type=int)
parser.add_argument('--debug', help='verbose',default=False, action='store_true')
args = parser.parse_args()
args.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(args.seed)))
print(args)

#class Imagenet(nn.Module):
#    def __init__(self, encoder):
#        super(Imagenet, self).__init__()
#        self.encoder = encoder
#        self.task = nn.Sequential(nn.Conv2d(512, 1000, kernel_size=1, stride=1)) # linearly project [2,2,512] features to [2,2,1000] classes
#
#    def forward(self, x):
#        fmap = self.encoder(x)
#        x = self.task(fmap)
#        return x

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

model = torch.load(args.model)
model.eval() # eval mode
if args.debug:
    torchinfo.summary(model, input_size=(1, 3, 700, 700))
device = torch.device('cpu')
model = model.to(device)
print('imagenet classifier model loaded')

with open('{}/imagenet/val/flist'.format(args.imagenet), 'r') as f:
    flist = f.readlines()
#flist = random.choices(flist,k=10000) # DEBUG
print('imagenet validation flist loaded',len(flist))

with open('{}/imagenet/devkit-1.0/data/ILSVRC2010_validation_ground_truth.txt'.format(args.imagenet), 'r') as f:
    vlist = f.readlines()
print('validation metadata loaded')

synlabel={}
labeltext={}
meta = scipy.io.loadmat('{}/imagenet/devkit-1.0/data/meta.mat'.format(args.imagenet))
synsets = meta['synsets']
for i,s in enumerate(synsets):
    synlabel[s[0][1][0]] = s[0][0][0][0]
    labeltext[s[0][0][0][0]] = s[0][2][0]
print('imagenet metadata loaded')

random.shuffle(flist)
acc=[]
if args.single:
    stride=1
if args.quad:
    stride=4
for i in range(0,len(flist),stride):
    if args.single:
        #fn = random.choice(flist)
        fn = flist[i]
        img = cv2.imread('{}/imagenet/val/{}'.format(args.imagenet,fn.rstrip()))
        img = cv2.resize(img,dsize=(700,700),interpolation=cv2.INTER_CUBIC)
    if args.quad:
        img = np.zeros([700,700,3]).astype(np.uint8)
        #s = random.choices(flist,k=4)
        s = flist[i:i+4]
        for j,fn in enumerate(s): #fn = 'n02487347_1956.JPEG'
            while True:
                img0 = cv2.imread('{}/imagenet/val/{}'.format(args.imagenet,fn.rstrip()))
                if img0 is None:
                    fn = random.choice(flist)
                else:
                    break
            img0 = cv2.resize(img0,dsize=(350,350),interpolation=cv2.INTER_CUBIC)
            x0 = 350*(j%2)
            y0 = 350*(j//2)
            img[y0:y0+350,x0:x0+350] = img0

    x = img.astype(np.float32)/255.
    x = np.expand_dims(x, axis=0)
    x = np.rollaxis(x,-1,1)
    x=torch.utils.data.default_convert(x)
    x = x.to(device)
    logits = model(x)
    if args.alt=='alt3':
        logits=logits[0]
    logits = logits.detach().numpy()
    if args.finetune:
        z = logits[0,:]
        z = z - np.max(z, axis=-1, keepdims=True)
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        prob = numerator / denominator
        #prob = np.exp(logits[0,:])/np.exp(logits[0,:]).sum()
        v = int(fn[-13:-6].lstrip('0'))-1
        gt = int(vlist[v].rstrip()) # ILSVRC2010_val_00050000.JPEG
        match=False
        for n in range(args.topn):
            pred = np.argsort(prob)[-1*(1+n)]
            if (pred+1)==gt:
                match=True
            if args.debug:
                print('i {:4d} n {:4d} prob {:.6f} pred {:6d} text {:40} gt {:6d} text {:40}'.format(
                    i,n,prob[pred],pred+1,labeltext[pred+1][0:40],gt,labeltext[gt][0:40]))
        acc.append(match)

    else:
        if args.debug:
            print(100*'-')
        for j in range(2):
            for k in range(2):
                prob = np.exp(logits[0,:,j,k])/np.exp(logits[0,:,j,k]).sum()
                if args.quad:
                    fn = s[j*2+k]
                v = int(fn[-13:-6].lstrip('0'))-1
                gt = int(vlist[v].rstrip()) # ILSVRC2010_val_00050000.JPEG
                match=False
                for n in range(args.topn):
                    pred = np.argsort(prob)[-1*(1+n)]
                    if (pred+1)==gt:
                        match=True
                    if args.debug:
                        print('i {:4d} j {:4d} k {:4d} n {:4d} prob {:.6f} pred {:6d} text {:40} gt {:6d} text {:40}'.format(
                            i,j,k,n,prob[pred],pred+1,labeltext[pred+1][0:40],gt,labeltext[gt][0:40]))
                if args.debug:
                    print(100*'-')
                acc.append(match)
    if args.debug:
        print()
        cv2.imshow('validation', img)
        k=cv2.waitKey(0)
        if k==120: # 'x'
            break
    else:
        if (i%1000)==0:
            print('i',i,'acc',np.mean(acc))
print('top {} accuracy {}'.format(args.topn,np.mean(acc)))

exit()

def generate_batch(args,flist,synlabel,labeltext):
    d = np.zeros([args.batch,700,700,3]).astype(np.uint8)
    l = np.zeros([args.batch,2,2]).astype(np.int64)
    for i in range(args.batch):
        s = random.choices(flist,k=4)
        while True:
            o = np.random.normal(350,100,size=2)
            if np.max(o)<650 and np.min(o)>=50:
                o=o.astype(int)
                break
        #o = np.random.randint(350-175,350+175)
        #f = random.choice([0,1])
        for j,fn in enumerate(s):
            #fn = 'n02487347_1956.JPEG'
            while True:
                img = cv2.imread('{}/imagenet/train/{}'.format(args.imagenet,fn.rstrip()))
                if img is None:
                    fn = random.choice(flist)
                else:
                    break
            #if f==0:
            #    if j==0:
            #        ulx = 0
            #        uly = 0
            #        lrx = o
            #        lry = o
            #    if j==1:
            #        ulx = o
            #        uly = 0
            #        lrx = 700
            #        lry = o
            #    if j==2:
            #        ulx = 0
            #        uly = o
            #        lrx = o
            #        lry = 700
            #    if j==3:
            #        ulx = o
            #        uly = o
            #        lrx = 700
            #        lry = 700
            #if f==1:
            #    if j==0:
            #        ulx = 0
            #        uly = 0
            #        lrx = o
            #        lry = 700-o
            #    if j==1:
            #        ulx = o
            #        uly = 0
            #        lrx = 700
            #        lry = 700-o
            #    if j==2:
            #        ulx = 0
            #        uly = 700-o
            #        lrx = o
            #        lry = 700
            #    if j==3:
            #        ulx = o
            #        uly = 700-o
            #        lrx = 700
            #        lry = 700

            if args.resize:
                if j==0:
                    ulx = 0
                    uly = 0
                    lrx = o[0]
                    lry = o[1]
                if j==1:
                    ulx = o[0]
                    uly = 0
                    lrx = 700
                    lry = o[1]
                if j==2:
                    ulx = 0
                    uly = o[1]
                    lrx = o[0]
                    lry = 700
                if j==3:
                    ulx = o[0]
                    uly = o[1]
                    lrx = 700
                    lry = 700
                img = cv2.resize(img,dsize=(lrx-ulx,lry-uly),interpolation=cv2.INTER_CUBIC)
                d[i,uly:lry,ulx:lrx] = img
            else:
                img = cv2.resize(img,dsize=(350,350),interpolation=cv2.INTER_CUBIC)
                x0 = 350*(j%2)
                y0 = 350*(j//2)
                d[i,y0:y0+350,x0:x0+350] = img

            #img = cv2.resize(img,dsize=(350,350),interpolation=cv2.INTER_LINEAR)
            #print('o',o,'f',f,'i',i,'j',j,'ulx',ulx,'uly',uly,'lrx',lrx,'lry',lry)
            #if random.choice([True,False]):
            #    img = cv2.flip(img,1) # horizontal flip
            #img = cv2.resize(img,dsize=(lrx-ulx,lry-uly),interpolation=cv2.INTER_LINEAR)
            ##    img = np.zeros_like(img)
            #x0 = 350*(j%2)
            #y0 = 350*(j//2)
            #d[i,y0:y0+350,x0:x0+350] = img
            #d[i,uly:lry,ulx:lrx] = img
            #print('fn',fn,fn[0:9])
            l[i,j//2,j%2] = synlabel[fn[0:9]]-1
            #print('j',j,'synlabel',synlabel[fn[0:9]],'labeltext',labeltext[synlabel[fn[0:9]]])

    #cv2.imshow('imagenet', cv2.resize(d[0],dsize=(dw,dh),interpolation=cv2.INTER_LINEAR)) 
    #cv2.imshow('imagenet', d[0])
    #print('l',l[0])
    #cv2.waitKey(1)
    d = d.astype(np.float32)/255.
    d = np.rollaxis(d,-1,1)
    return d,l





#with open(args.log, 'a') as f:
#    print(args,file=f)

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

encoder = torch.load('{}.pt'.format(args.encoder))
print('image encoder model loaded')

class Imagenet(nn.Module):
    def __init__(self, encoder):
        super(Imagenet, self).__init__()
        self.encoder = encoder
        self.task = nn.Sequential(nn.Conv2d(512, 1000, kernel_size=1, stride=1)) # linearly project [2,2,512] features to [2,2,1000] classes

    def forward(self, x):
        fmap = self.encoder(x)
        x = self.task(fmap)
        return x

def generate_batch(args,flist,synlabel,labeltext):
    d = np.zeros([args.batch,700,700,3]).astype(np.uint8)
    l = np.zeros([args.batch,2,2]).astype(np.int64)
    for i in range(args.batch):
        s = random.choices(flist,k=4)
        while True:
            o = np.random.normal(350,100,size=2)
            if np.max(o)<650 and np.min(o)>=50:
                o=o.astype(int)
                break
        #o = np.random.randint(350-175,350+175)
        #f = random.choice([0,1])
        for j,fn in enumerate(s):
            #fn = 'n02487347_1956.JPEG'
            while True:
                img = cv2.imread('{}/imagenet/train/{}'.format(args.imagenet,fn.rstrip()))
                if img is None:
                    fn = random.choice(flist)
                else:
                    break
            #if f==0:
            #    if j==0:
            #        ulx = 0
            #        uly = 0
            #        lrx = o
            #        lry = o
            #    if j==1:
            #        ulx = o
            #        uly = 0
            #        lrx = 700
            #        lry = o
            #    if j==2:
            #        ulx = 0
            #        uly = o
            #        lrx = o
            #        lry = 700
            #    if j==3:
            #        ulx = o
            #        uly = o
            #        lrx = 700
            #        lry = 700
            #if f==1:
            #    if j==0:
            #        ulx = 0
            #        uly = 0
            #        lrx = o
            #        lry = 700-o
            #    if j==1:
            #        ulx = o
            #        uly = 0
            #        lrx = 700
            #        lry = 700-o
            #    if j==2:
            #        ulx = 0
            #        uly = 700-o
            #        lrx = o
            #        lry = 700
            #    if j==3:
            #        ulx = o
            #        uly = 700-o
            #        lrx = 700
            #        lry = 700

            if j==0:
                ulx = 0
                uly = 0
                lrx = o[0]
                lry = o[1]
            if j==1:
                ulx = o[0]
                uly = 0
                lrx = 700
                lry = o[1]
            if j==2:
                ulx = 0
                uly = o[1]
                lrx = o[0]
                lry = 700
            if j==3:
                ulx = o[0]
                uly = o[1]
                lrx = 700
                lry = 700

            #img = cv2.resize(img,dsize=(350,350),interpolation=cv2.INTER_LINEAR)
            #print('o',o,'f',f,'i',i,'j',j,'ulx',ulx,'uly',uly,'lrx',lrx,'lry',lry)
            if random.choice([True,False]):
                img = cv2.flip(img,1) # horizontal flip
            img = cv2.resize(img,dsize=(lrx-ulx,lry-uly),interpolation=cv2.INTER_CUBIC)
            ##    img = np.zeros_like(img)
            #x0 = 350*(j%2)
            #y0 = 350*(j//2)
            #d[i,y0:y0+350,x0:x0+350] = img
            d[i,uly:lry,ulx:lrx] = img
            #print('fn',fn,fn[0:9])
            l[i,j//2,j%2] = synlabel[fn[0:9]]-1
            #print('j',j,'synlabel',synlabel[fn[0:9]],'labeltext',labeltext[synlabel[fn[0:9]]])

    #cv2.imshow('imagenet', cv2.resize(d[0],dsize=(dw,dh),interpolation=cv2.INTER_LINEAR)) 
    cv2.imshow('imagenet', d[0])
    #print('l',l[0])
    cv2.waitKey(1)
    d = d.astype(np.float32)/255.
    d = np.rollaxis(d,-1,1)
    return d,l

model = Imagenet(encoder)
torchinfo.summary(model, input_size=(1, 3, 700, 700))
device = torch.device('cuda')
model = model.to(device)

# Loss and optimizer
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
#torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for i in range(args.train):
    (x,y)=generate_batch(args,flist,synlabel,labeltext)
    x=torch.utils.data.default_convert(x)
    y=torch.utils.data.default_convert(y)
    x = x.to(device)
    y = y.to(device)
        
    # train batch
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute gradient
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    #print ('batch [{}/{}] loss {:.4f} grad {:.4f}'.format(i+1, args.train, loss.item(),total_norm))
    s = 'BATCH {:12d} wall {} loss {:12.6f} grad {:12.6f} lr {:.9f}'.format(i+1,datetime.datetime.now(),loss.item(),total_norm,args.lr)
    print(s)
#    with open(args.log, 'a') as f:
#        print(s,file=f)
    if (i%1000)==0:
        torch.save(encoder, '{}.pt'.format(args.encoder))
        torch.save(model, 'imagenet.pt')
