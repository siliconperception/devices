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
parser.add_argument('--resize', help='resize scale',default=0.1, type=float)
parser.add_argument('--imagenet', help='imagenet dataset base directory',default='../../')
parser.add_argument('--scratch', help='start training from random weights',default=False, action='store_true')
parser.add_argument('--checkpoint', help='save timestamped checkpoint every 100000 batches',default=False, action='store_true')
parser.add_argument('--save', help='save temporary checkpoint every 1000 batches',default=False, action='store_true')
parser.add_argument('--encoder', help='encoder model name',default='ie120-050-240')
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--train', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--batch', help='batch size',default=10, type=int)
parser.add_argument('--lr', help='learning rate',default=0.0005, type=float)
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
    for i in range(args.batch):
        rf=[] # 2x2 receptive fields
        rf.append(Rectangle((700//2)*0, (700//2)*0, (700//2)*1, (700//2)*1))
        rf.append(Rectangle((700//2)*1, (700//2)*0, (700//2)*2, (700//2)*1))
        rf.append(Rectangle((700//2)*0, (700//2)*1, (700//2)*1, (700//2)*2))
        rf.append(Rectangle((700//2)*1, (700//2)*1, (700//2)*2, (700//2)*2))

        sc = random.choice(['1x1','2x2','3x3'])
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
            ir.append(Rectangle(0, 0, x0, y0))
            ir.append(Rectangle(x0, 0, x1, y0))
            ir.append(Rectangle(x1, 0, 700, y0))
            ir.append(Rectangle(0, y0, x0, y1))
            ir.append(Rectangle(x0, y0, x1, y1))
            ir.append(Rectangle(x1, y0, 700, y1))
            ir.append(Rectangle(0, y1, x0, 700))
            ir.append(Rectangle(x0, y1, x1, 700))
            ir.append(Rectangle(x1, y1, 700, 700))

        # populate d[] with len(ir) images
        il=[] # image labels corresponding to ir[]
        for r in ir:
            while True:
                fn = random.choice(flist)
                img = cv2.imread('{}/imagenet/train/{}'.format(args.imagenet,fn.rstrip()))
                if img is not None:
                    break

            #d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = cv2.resize(img,dsize=(r.xmax-r.xmin,r.ymax-r.ymin),interpolation=cv2.INTER_LINEAR)
            d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = cv2.resize(img,dsize=(r.xmax-r.xmin,r.ymax-r.ymin),interpolation=cv2.INTER_CUBIC)
            il.append(synlabel[fn[0:9]]-1)

        # for each rf, populate l[] with len(ir) [0,1] probabilities based on overlap between rf and ir rectangles
        for j,r0 in enumerate(rf):
            for k,r1 in enumerate(ir):
                dx = min(r0.xmax, r1.xmax) - max(r0.xmin, r1.xmin)
                dy = min(r0.ymax, r1.ymax) - max(r0.ymin, r1.ymin)
                if (dx>=0) and (dy>=0):
                    if (r0.xmax-r0.xmin)*(r0.ymax-r0.ymin) > (r1.xmax-r1.xmin)*(r1.ymax-r1.ymin):
                        overlap = (dx*dy)/((r0.xmax-r0.xmin)*(r0.ymax-r0.ymin)) # fraction of r1 which overlaps with r0
                    else:
                        overlap = (dx*dy)/((r1.xmax-r1.xmin)*(r1.ymax-r1.ymin)) # fraction of r0 which overlaps with r1
                    l[i,j//2,j%2,il[k]] = overlap

    d = d.astype(np.float32)/255.
    d = np.rollaxis(d,-1,1)
    l = np.rollaxis(l,-1,1)
    return d,l

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
            (d,l) = generate_batch(self.args,self.flist,self.synlabel,self.labeltext)
            q.put((d,l))

model = Imagenet.Imagenet(encoder)
if args.debug:
    torchinfo.summary(model, input_size=(1, 3, 700, 700))
device = torch.device('cuda')
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

q = queue.Queue(maxsize=100)
for _ in range(20):
    w = Worker(q,args,flist,synlabel,labeltext)
    w.daemon = True
    w.start()

i=0
larr=[]
garr=[]
while i<args.train:
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
    s = 'BATCH {:12d} wall {} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f}'.format(
        i+1,datetime.datetime.now(),loss.item(),np.mean(larr[-100:]),total_norm,np.mean(garr[-100:]))
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
