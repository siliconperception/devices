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
from collections import namedtuple

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--centercrop', help='crop to square',default=False, action='store_true')
parser.add_argument('--x1', help='include 1x1 examples',default=False, action='store_true')
parser.add_argument('--x2', help='include 2x2 examples',default=False, action='store_true')
parser.add_argument('--x3', help='include 3x3 examples',default=False, action='store_true')
parser.add_argument('--maxpool', help='test single image imagenet classifier by adding quadrant predictions',default=False, action='store_true')
parser.add_argument('--finetune', help='test single image finetuned imagenet classifier',default=False, action='store_true')
parser.add_argument('--imagenet', help='imagenet dataset base directory',default='../../')
parser.add_argument('--alt', help='pretrain arch',default='alt3')
parser.add_argument('--topn', help='include top n predictions in accuracy calculation',default=5, type=int)
parser.add_argument('--model', help='pytorch model filename',default='pretrain.pt')
parser.add_argument('--single', help='test single image',default=False, action='store_true')
parser.add_argument('--quad', help='test 4-quadrant image',default=False, action='store_true')
#parser.add_argument('--encoder', help='encoder model name',default='ie120-050-240')
#parser.add_argument('--log', help='log file name',default='log')
parser.add_argument('--nbatch', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--batch', help='batch size',default=3, type=int)
#parser.add_argument('--lr', help='learning rate',default=0.0001, type=float)
#parser.add_argument('--show', help='display batches',default=False, action='store_true')
parser.add_argument('--seed', help='random seed',default=None, type=int)
parser.add_argument('--debug', help='verbose',default=False, action='store_true')
args = parser.parse_args()
args.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(args.seed)))
print(args)

def generate_batch(args,flist,synlabel,labeltext,vlist):
    d = np.zeros([args.batch,700,700,3]).astype(np.uint8)
    l = np.zeros([args.batch,2,2,1000]).astype(float) # class probabilities for 2x2 receptive fields

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    choices = []
    if args.x1:
        choices.append('1x1')
    if args.x2:
        choices.append('2x2')
    if args.x3:
        choices.append('3x3')

    for i in range(args.batch):
        rf=[] # 2x2 receptive fields
        rf.append(Rectangle((700//2)*0, (700//2)*0, (700//2)*1, (700//2)*1))
        rf.append(Rectangle((700//2)*1, (700//2)*0, (700//2)*2, (700//2)*1))
        rf.append(Rectangle((700//2)*0, (700//2)*1, (700//2)*1, (700//2)*2))
        rf.append(Rectangle((700//2)*1, (700//2)*1, (700//2)*2, (700//2)*2))

        sc = random.choice(choices)
        ir=[] # image rectangles
        if sc=='1x1':
            ir.append(Rectangle(0, 0, 700, 700))
        if sc=='2x2':
            s = (700*args.resize)//2
            x0 = np.random.randint((700//2)-s,(700//2)+s+1)
            y0 = np.random.randint((700//2)-s,(700//2)+s+1)
            ir.append(Rectangle(0, 0, x0, y0))
            ir.append(Rectangle(x0, 0, 700, y0))
            ir.append(Rectangle(0, y0, x0, 700))
            ir.append(Rectangle(x0, y0, 700, 700))
        if sc=='3x3':
            s = (700*args.resize)//3
            x0 = np.random.randint(1*(700//3)-s,1*(700//3)+s+1)
            x1 = np.random.randint(2*(700//3)-s,2*(700//3)+s+1)
            y0 = np.random.randint(1*(700//3)-s,1*(700//3)+s+1)
            y1 = np.random.randint(2*(700//3)-s,2*(700//3)+s+1)
            ir.append(Rectangle(0, 0, x0, y0)) # 0
            ir.append(Rectangle(x0, 0, x1, y0)) # 1
            ir.append(Rectangle(x1, 0, 700, y0)) # 2
            ir.append(Rectangle(0, y0, x0, y1)) # 3
            ir.append(Rectangle(x0, y0, x1, y1)) # 4
            ir.append(Rectangle(x1, y0, 700, y1)) # 5
            ir.append(Rectangle(0, y1, x0, 700)) # 6
            ir.append(Rectangle(x0, y1, x1, 700)) # 7
            ir.append(Rectangle(x1, y1, 700, 700)) # 8

        # populate d[] with len(ir) images
        il=[] # image labels corresponding to ir[]
        for r in ir:
            while True:
                fn = random.choice(flist)
                img = cv2.imread('{}/imagenet/val/{}'.format(args.imagenet,fn.rstrip()))
                if img is not None:
                    break
            if args.centercrop:
                side = min(img.shape[0],img.shape[1])
                img = img[img.shape[0]//2-side//2:img.shape[0]//2+side//2,img.shape[1]//2-side//2:img.shape[1]//2+side//2]
            #d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = cv2.resize(img,dsize=(r.xmax-r.xmin,r.ymax-r.ymin),interpolation=cv2.INTER_LINEAR)
            d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = cv2.resize(img,dsize=(r.xmax-r.xmin,r.ymax-r.ymin),interpolation=cv2.INTER_CUBIC)
            v = int(fn[-13:-6].lstrip('0'))-1
            gt_zb = int(vlist[v].rstrip())-1 # ILSVRC2010_val_00050000.JPEG
            #print('fn',fn,'v',v,'gt',gt)
            il.append(gt_zb)

        # for each rf, populate l[] with len(ir) [0,1] probabilities based on overlap between rf and ir rectangles
        if sc=='1x1':
            l[i,0,0,il[0]] = 0.25
            l[i,0,1,il[0]] = 0.25
            l[i,1,0,il[0]] = 0.25
            l[i,1,1,il[0]] = 0.25
        if sc=='2x2':
            l[i,0,0,il[0]] = 1.0
            l[i,0,1,il[1]] = 1.0
            l[i,1,0,il[2]] = 1.0
            l[i,1,1,il[3]] = 1.0
        if sc=='3x3':
            l[i,0,0,il[0]] = 1.0
            l[i,0,0,il[1]] = 0.5
            l[i,0,0,il[3]] = 0.5
            l[i,0,0,il[4]] = 0.25
            l[i,0,1,il[2]] = 1.0
            l[i,0,1,il[1]] = 0.5
            l[i,0,1,il[5]] = 0.5
            l[i,0,1,il[4]] = 0.25
            l[i,1,0,il[6]] = 1.0
            l[i,1,0,il[3]] = 0.5
            l[i,1,0,il[7]] = 0.5
            l[i,1,0,il[4]] = 0.25
            l[i,1,1,il[8]] = 1.0
            l[i,1,1,il[5]] = 0.5
            l[i,1,1,il[7]] = 0.5
            l[i,1,1,il[4]] = 0.25

    d = d.astype(np.float32)/255.
    d = np.rollaxis(d,-1,1)
    l = np.rollaxis(l,-1,1)
    return d,l


device = torch.device('cuda')
model = torch.load(args.model,map_location=device)
model.eval() # eval mode
if args.debug:
    torchinfo.summary(model, input_size=(1, 3, 700, 700))
#device = torch.device('cpu')
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

acc=[]
for i in range(args.nbatch):
    (d,l) = generate_batch(args,flist,synlabel,labeltext,vlist)
    d=torch.utils.data.default_convert(d).to(device)
    logits = model(d)
    #print('logits',logits.shape)
    if args.quad:
        prob = torch.nn.functional.softmax(logits,dim=-3)
    if args.single:
        prob = torch.nn.functional.softmax(logits,dim=-3)
        prob = torch.reshape(prob,[-1,1000])
        l = np.sum(l,axis=(-1,-2))
    #print('prob',prob.shape,'l',l.shape)
    
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
    print('i',i,'acc',np.mean(acc),'acc0',np.mean(acc0),'n',len(acc),len(acc0))




exit()
# -------------------------------------------------------------------------------------------------------------------------------
# load encoder model, train task to predict imagenet classes for four quadrants
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
from collections import namedtuple

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--centercrop', help='crop to square',default=False, action='store_true')
parser.add_argument('--step', help='LR scheduler batches per step',default=10000, type=int)
parser.add_argument('--end_factor', help='LR linear schedule parameter',default=1./5, type=float)
parser.add_argument('--total_iters', help='LR linear schedule parameter',default=10, type=float)
parser.add_argument('--x1', help='include 1x1 examples',default=False, action='store_true')
parser.add_argument('--x2', help='include 2x2 examples',default=False, action='store_true')
parser.add_argument('--x3', help='include 3x3 examples',default=False, action='store_true')
parser.add_argument('--workers', help='number of threads for batch generation',default=20, type=int)
parser.add_argument('--resize', help='resize scale',default=0.0, type=float)
parser.add_argument('--imagenet', help='imagenet dataset base directory',default='../../')
parser.add_argument('--scratch', help='start training from random weights',default=False, action='store_true')
parser.add_argument('--checkpoint', help='save timestamped checkpoint every 100000 batches',default=False, action='store_true')
parser.add_argument('--save', help='save temporary checkpoint every 1000 batches',default=False, action='store_true')
parser.add_argument('--encoder', help='encoder model name',default='ie120-050-240')
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--train', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--batch', help='batch size',default=32, type=int)
parser.add_argument('--lr', help='initial learning rate',default=0.0005, type=float)
#parser.add_argument('--gamma', help='LR decay',default=0.9, type=float)
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
print('imagenet metadata loaded')

with open('{}/imagenet/train/flist'.format(args.imagenet), 'r') as f:
    flist = f.readlines()
#flist = random.choices(flist,k=10) # DEBUG
print('imagenet flist loaded',len(flist))

if args.scratch:
    encoder = IE120.IE120(args.encoder)
    print('image encoder model initialized')
else:
    encoder = torch.load('{}.pt'.format(args.encoder))
    print('image encoder model loaded')

encoder.train() # train mode

def generate_batch(args,flist,synlabel,labeltext):
    d = np.zeros([args.batch,700,700,3]).astype(np.uint8)
    l = np.zeros([args.batch,2,2,1000]).astype(float) # class probabilities for 2x2 receptive fields

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    choices = []
    if args.x1:
        choices.append('1x1')
    if args.x2:
        choices.append('2x2')
    if args.x3:
        choices.append('3x3')

    for i in range(args.batch):
        rf=[] # 2x2 receptive fields
        rf.append(Rectangle((700//2)*0, (700//2)*0, (700//2)*1, (700//2)*1))
        rf.append(Rectangle((700//2)*1, (700//2)*0, (700//2)*2, (700//2)*1))
        rf.append(Rectangle((700//2)*0, (700//2)*1, (700//2)*1, (700//2)*2))
        rf.append(Rectangle((700//2)*1, (700//2)*1, (700//2)*2, (700//2)*2))

        sc = random.choice(choices)
        ir=[] # image rectangles
        if sc=='1x1':
            ir.append(Rectangle(0, 0, 700, 700))
        if sc=='2x2':
            s = (700*args.resize)//2
            x0 = np.random.randint((700//2)-s,(700//2)+s+1)
            y0 = np.random.randint((700//2)-s,(700//2)+s+1)
            ir.append(Rectangle(0, 0, x0, y0))
            ir.append(Rectangle(x0, 0, 700, y0))
            ir.append(Rectangle(0, y0, x0, 700))
            ir.append(Rectangle(x0, y0, 700, 700))
        if sc=='3x3':
            s = (700*args.resize)//3
            x0 = np.random.randint(1*(700//3)-s,1*(700//3)+s+1)
            x1 = np.random.randint(2*(700//3)-s,2*(700//3)+s+1)
            y0 = np.random.randint(1*(700//3)-s,1*(700//3)+s+1)
            y1 = np.random.randint(2*(700//3)-s,2*(700//3)+s+1)
            ir.append(Rectangle(0, 0, x0, y0)) # 0
            ir.append(Rectangle(x0, 0, x1, y0)) # 1
            ir.append(Rectangle(x1, 0, 700, y0)) # 2
            ir.append(Rectangle(0, y0, x0, y1)) # 3
            ir.append(Rectangle(x0, y0, x1, y1)) # 4
            ir.append(Rectangle(x1, y0, 700, y1)) # 5
            ir.append(Rectangle(0, y1, x0, 700)) # 6
            ir.append(Rectangle(x0, y1, x1, 700)) # 7
            ir.append(Rectangle(x1, y1, 700, 700)) # 8

        # populate d[] with len(ir) images
        il=[] # image labels corresponding to ir[]
        for r in ir:
            while True:
                fn = random.choice(flist)
                img = cv2.imread('{}/imagenet/train/{}'.format(args.imagenet,fn.rstrip()))
                if img is not None:
                    break
            if args.centercrop:
                side = min(img.shape[0],img.shape[1])
                img = img[img.shape[0]//2-side//2:img.shape[0]//2+side//2,img.shape[1]//2-side//2:img.shape[1]//2+side//2]
            #d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = cv2.resize(img,dsize=(r.xmax-r.xmin,r.ymax-r.ymin),interpolation=cv2.INTER_LINEAR)
            d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = cv2.resize(img,dsize=(r.xmax-r.xmin,r.ymax-r.ymin),interpolation=cv2.INTER_CUBIC)
            il.append(synlabel[fn[0:9]]-1)

        # for each rf, populate l[] with len(ir) [0,1] probabilities based on overlap between rf and ir rectangles
        if sc=='1x1':
            l[i,0,0,il[0]] = 0.25
            l[i,0,1,il[0]] = 0.25
            l[i,1,0,il[0]] = 0.25
            l[i,1,1,il[0]] = 0.25
        if sc=='2x2':
            l[i,0,0,il[0]] = 1.0
            l[i,0,1,il[1]] = 1.0
            l[i,1,0,il[2]] = 1.0
            l[i,1,1,il[3]] = 1.0
        if sc=='3x3':
            l[i,0,0,il[0]] = 1.0
            l[i,0,0,il[1]] = 0.5
            l[i,0,0,il[3]] = 0.5
            l[i,0,0,il[4]] = 0.25
            l[i,0,1,il[2]] = 1.0
            l[i,0,1,il[1]] = 0.5
            l[i,0,1,il[5]] = 0.5
            l[i,0,1,il[4]] = 0.25
            l[i,1,0,il[6]] = 1.0
            l[i,1,0,il[3]] = 0.5
            l[i,1,0,il[7]] = 0.5
            l[i,1,0,il[4]] = 0.25
            l[i,1,1,il[8]] = 1.0
            l[i,1,1,il[5]] = 0.5
            l[i,1,1,il[7]] = 0.5
            l[i,1,1,il[4]] = 0.25

#        if sc=='1x1':
#            l[i,0,0,il[0]] = 0.25
#            l[i,0,1,il[0]] = 0.25
#            l[i,1,0,il[0]] = 0.25
#            l[i,1,1,il[0]] = 0.25
#        if sc=='2x2':
#            l[i,0,0,il[0]] = 1.0 * (((ir[0].xmax-ir[0].xmin)*(ir[0].ymax-ir[0].ymin)) / ((700/2)*(700/2)))
#            l[i,0,1,il[1]] = 1.0 * (((ir[1].xmax-ir[1].xmin)*(ir[1].ymax-ir[1].ymin)) / ((700/2)*(700/2)))
#            l[i,1,0,il[2]] = 1.0 * (((ir[2].xmax-ir[2].xmin)*(ir[2].ymax-ir[2].ymin)) / ((700/2)*(700/2)))
#            l[i,1,1,il[3]] = 1.0 * (((ir[3].xmax-ir[3].xmin)*(ir[3].ymax-ir[3].ymin)) / ((700/2)*(700/2)))
#        if sc=='3x3':
#            l[i,0,0,il[0]] = 1.0 * (((ir[0].xmax-ir[0].xmin)*(ir[0].ymax-ir[0].ymin)) / ((700/3)*(700/3)))
#            l[i,0,0,il[1]] = 0.5 * (((ir[1].xmax-ir[1].xmin)*(ir[1].ymax-ir[1].ymin)) / ((700/3)*(700/3)))
#            l[i,0,0,il[3]] = 0.5 * (((ir[3].xmax-ir[3].xmin)*(ir[3].ymax-ir[3].ymin)) / ((700/3)*(700/3)))
#            l[i,0,0,il[4]] = 0.25 * (((ir[4].xmax-ir[4].xmin)*(ir[4].ymax-ir[4].ymin)) / ((700/3)*(700/3)))
#            l[i,0,1,il[2]] = 1.0 * (((ir[2].xmax-ir[2].xmin)*(ir[2].ymax-ir[2].ymin)) / ((700/3)*(700/3)))
#            l[i,0,1,il[1]] = 0.5 * (((ir[1].xmax-ir[1].xmin)*(ir[1].ymax-ir[1].ymin)) / ((700/3)*(700/3)))
#            l[i,0,1,il[5]] = 0.5 * (((ir[5].xmax-ir[5].xmin)*(ir[5].ymax-ir[5].ymin)) / ((700/3)*(700/3)))
#            l[i,0,1,il[4]] = 0.25 * (((ir[4].xmax-ir[4].xmin)*(ir[4].ymax-ir[4].ymin)) / ((700/3)*(700/3)))
#            l[i,1,0,il[6]] = 1.0 * (((ir[6].xmax-ir[6].xmin)*(ir[6].ymax-ir[6].ymin)) / ((700/3)*(700/3)))
#            l[i,1,0,il[3]] = 0.5 * (((ir[3].xmax-ir[3].xmin)*(ir[3].ymax-ir[3].ymin)) / ((700/3)*(700/3)))
#            l[i,1,0,il[7]] = 0.5 * (((ir[7].xmax-ir[7].xmin)*(ir[7].ymax-ir[7].ymin)) / ((700/3)*(700/3)))
#            l[i,1,0,il[4]] = 0.25 * (((ir[4].xmax-ir[4].xmin)*(ir[4].ymax-ir[4].ymin)) / ((700/3)*(700/3)))
#            l[i,1,1,il[8]] = 1.0 * (((ir[8].xmax-ir[8].xmin)*(ir[8].ymax-ir[8].ymin)) / ((700/3)*(700/3)))
#            l[i,1,1,il[5]] = 0.5 * (((ir[5].xmax-ir[5].xmin)*(ir[5].ymax-ir[5].ymin)) / ((700/3)*(700/3)))
#            l[i,1,1,il[7]] = 0.5 * (((ir[7].xmax-ir[7].xmin)*(ir[7].ymax-ir[7].ymin)) / ((700/3)*(700/3)))
#            l[i,1,1,il[4]] = 0.25 * (((ir[4].xmax-ir[4].xmin)*(ir[4].ymax-ir[4].ymin)) / ((700/3)*(700/3)))

        #for j,r0 in enumerate(rf):
        #    for k,r1 in enumerate(ir):
        #        dx = min(r0.xmax, r1.xmax) - max(r0.xmin, r1.xmin)
        #        dy = min(r0.ymax, r1.ymax) - max(r0.ymin, r1.ymin)
        #        if (dx>=0) and (dy>=0):
        #            if (r0.xmax-r0.xmin)*(r0.ymax-r0.ymin) > (r1.xmax-r1.xmin)*(r1.ymax-r1.ymin):
        #                overlap = (dx*dy)/((r0.xmax-r0.xmin)*(r0.ymax-r0.ymin)) # fraction of r1 which overlaps with r0
        #            else:
        #                overlap = (dx*dy)/((r1.xmax-r1.xmin)*(r1.ymax-r1.ymin)) # fraction of r0 which overlaps with r1
        #            l[i,j//2,j%2,il[k]] = overlap

    d = d.astype(np.float32)/255.
    d = np.rollaxis(d,-1,1)
    l = np.rollaxis(l,-1,1)
    return d,l

#class Worker(threading.Thread):
#    def __init__(self, q, args, flist, synlabel, labeltext):
#        self.q = q
#        self.args = args
#        self.flist = flist
#        self.synlabel = synlabel
#        self.labeltext = labeltext
#        self.running = True
#        super().__init__()
#    def run(self):
#        while self.running:
#            (d,l) = generate_batch(self.args,self.flist,self.synlabel,self.labeltext)
#            q.put((d,l))

model = Imagenet.Imagenet(encoder)
if args.debug:
    torchinfo.summary(model, input_size=(1, 3, 700, 700))
device = torch.device('cuda')
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=args.end_factor, total_iters=args.total_iters)

def worker(stop, q, args, flist, synlabel, labeltext):
    while not stop.is_set():
        (d,l) = generate_batch(args,flist,synlabel,labeltext)
        q.put((d,l))

q = queue.Queue(maxsize=args.workers)
stop = threading.Event()
stop.clear()
workers=[]
for _ in range(args.workers):
    #w = Worker(q,args,flist,synlabel,labeltext)
    #w.daemon = True
    w = threading.Thread(target=worker, args=[stop,q,args,flist,synlabel,labeltext], daemon=False)
    w.start()
    workers.append(w)

i=1
larr=[]
garr=[]
while i<1+args.train:
    (x0,y0)=q.get()

    x=torch.utils.data.default_convert(x0)
    x = x.to(device)
    y=torch.utils.data.default_convert(y0)
    y = y.to(device)
     
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i%args.step)==0:
        scheduler.step()

    # compute gradient
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # print batch statistics
    larr.append(loss.item())
    garr.append(total_norm)
    lr = optimizer.param_groups[0]['lr']
    s = 'BATCH {:12d} wall {} lr {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f}'.format(
        i,datetime.datetime.now(),lr,args.batch,loss.item(),np.mean(larr[-100:]),total_norm,np.mean(garr[-100:]))
    #s = 'BATCH {:12d} wall {} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f}'.format(
    #    i+1,datetime.datetime.now(),loss.item(),np.mean(larr[-100:]),total_norm,np.mean(garr[-100:]))
    print(s)
    with open(args.log, 'a') as f:
        print(s,file=f)
    
    if args.save and (i%1000)==0:
        torch.save(encoder, '{}.pt'.format(args.encoder))
        torch.save(model, 'pretrain.pt')

    if args.checkpoint and (i%100000)==0:
            torch.save(encoder, 'checkpoint/{}.{}.pt'.format(args.encoder,args.date))
            torch.save(model, 'checkpoint/pretrain.{}.pt'.format(args.date))

    if args.show and (i%1)==0:
        img = x0[0]*255
        img = img.astype(np.uint8)
        img = np.swapaxes(img,0,-1)
        img = np.swapaxes(img,0,1)
        cv2.imshow('imagenet', img)
        cv2.waitKey(1)

    i+=1
#print('DRAINING QUEUE')
#while not q.empty():
#    q.get()
print('STOPPING WORKERS')
stop.set()
for w in workers:
    while not q.empty():
        q.get()
    w.join()
print('EXIT MAIN')
# -------------------------------------------------------------------------------------------------------------------------------

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
        side = min(img.shape[0],img.shape[1])
        img = img[img.shape[0]//2-side//2:img.shape[0]//2+side//2,img.shape[1]//2-side//2:img.shape[1]//2+side//2]
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
            side = min(img.shape[0],img.shape[1])
            img = img[img.shape[0]//2-side//2:img.shape[0]//2+side//2,img.shape[1]//2-side//2:img.shape[1]//2+side//2]
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
        #z = logits[0,:]
        #z = z - np.max(z, axis=-1, keepdims=True)
        #numerator = np.exp(z)
        #denominator = np.sum(numerator, axis=-1, keepdims=True)
        #prob = numerator / denominator
        prob = np.exp(logits[0,:])/np.exp(logits[0,:]).sum()
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
        mp = np.zeros(1000)
        for j in range(2):
            for k in range(2):
                prob = np.exp(logits[0,:,j,k])/np.exp(logits[0,:,j,k]).sum()
                mp += prob
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
                if not args.maxpool:
                    acc.append(match)
    if args.maxpool:
        match=False
        for n in range(args.topn):
            pred = np.argsort(prob)[-1*(1+n)]
            if (pred+1)==gt:
                match=True
        acc.append(match)

    if args.debug:
        print()
        cv2.imshow('validation', img)
        k=cv2.waitKey(0)
        if k==120: # 'x'
            break
    else:
        if (i%100)==0:
            print('i',i,'acc',np.mean(acc),'n',len(acc))
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
device = torch.device('cpu')
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
