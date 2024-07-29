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
parser.add_argument('--freeze', help='freeze encoder weights',default=False, action='store_true')
parser.add_argument('--roi', help='input image x/y size',default=768, type=int)
parser.add_argument('--alt', help='encoder model alt type',default='alt2')
parser.add_argument('--centercrop', help='crop to square',default=False, action='store_true')
parser.add_argument('--step', help='LR scheduler batches per step',default=1000000000, type=int)
parser.add_argument('--end_factor', help='LR linear schedule parameter',default=1./50, type=float)
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
    encoder = torch.load('{}'.format(args.encoder))
    print('image encoder model loaded')

if args.freeze:
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval() # eval mode
    print('image encoder model frozen')

else:
    encoder.train() # train mode

def generate_batch(args,flist,synlabel,labeltext):
    d = np.zeros([args.batch,args.roi,args.roi,3]).astype(np.uint8)
    #l = np.zeros([args.batch,2,2,1000]).astype(float) # class probabilities for 2x2 receptive fields
    l0 = np.zeros([args.batch,1000,3,3]).astype(float) # class probabilities 3x3 feature map
    l1 = np.zeros([args.batch,1000,2,2]).astype(float) # class probabilities 2x2 feature map
    l2 = np.zeros([args.batch,1000,1,1]).astype(float) # class probabilities 1x1 feature map

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    choices = []
    if args.x1:
        choices.extend(1*['1x1'])
    if args.x2:
        choices.extend(4*['2x2'])
    if args.x3:
        choices.extend(9*['3x3'])

    for i in range(args.batch):
        #rf=[] # 2x2 receptive fields
        #rf.append(Rectangle((700//2)*0, (700//2)*0, (700//2)*1, (700//2)*1))
        #rf.append(Rectangle((700//2)*1, (700//2)*0, (700//2)*2, (700//2)*1))
        #rf.append(Rectangle((700//2)*0, (700//2)*1, (700//2)*1, (700//2)*2))
        #rf.append(Rectangle((700//2)*1, (700//2)*1, (700//2)*2, (700//2)*2))

        sc = random.choice(choices)
        ir=[] # image rectangles
        roi=args.roi
        if sc=='1x1':
            ir.append(Rectangle(0, 0, roi, roi))
        if sc=='2x2':
            s = (roi*args.resize)//2
            x0 = np.random.randint((roi//2)-s,(roi//2)+s+1)
            y0 = np.random.randint((roi//2)-s,(roi//2)+s+1)
            ir.append(Rectangle(0, 0, x0, y0))
            ir.append(Rectangle(x0, 0, roi, y0))
            ir.append(Rectangle(0, y0, x0, roi))
            ir.append(Rectangle(x0, y0, roi, roi))
        if sc=='3x3':
            s = (roi*args.resize)//3
            x0 = np.random.randint(1*(roi//3)-s,1*(roi//3)+s+1)
            x1 = np.random.randint(2*(roi//3)-s,2*(roi//3)+s+1)
            y0 = np.random.randint(1*(roi//3)-s,1*(roi//3)+s+1)
            y1 = np.random.randint(2*(roi//3)-s,2*(roi//3)+s+1)
            ir.append(Rectangle(0, 0, x0, y0)) # 0
            ir.append(Rectangle(x0, 0, x1, y0)) # 1
            ir.append(Rectangle(x1, 0, roi, y0)) # 2
            ir.append(Rectangle(0, y0, x0, y1)) # 3
            ir.append(Rectangle(x0, y0, x1, y1)) # 4
            ir.append(Rectangle(x1, y0, roi, y1)) # 5
            ir.append(Rectangle(0, y1, x0, roi)) # 6
            ir.append(Rectangle(x0, y1, x1, roi)) # 7
            ir.append(Rectangle(x1, y1, roi, roi)) # 8

        # populate d[] with len(ir) images
        il=[] # image labels corresponding to ir[]
        for r in ir:
            while True:
                fn = random.choice(flist)
                img = cv2.imread('{}/imagenet/train/{}'.format(args.imagenet,fn.rstrip()))
                if img is not None:
                    break
            if args.centercrop:
                side1 = min(img.shape[0],img.shape[1])
                img = img[img.shape[0]//2-side1//2:img.shape[0]//2+side1//2,img.shape[1]//2-side1//2:img.shape[1]//2+side1//2]
                #print('img',img.shape)
                sx = r.xmax-r.xmin
                sy = r.ymax-r.ymin
                m2 = max(sx,sy)
                img = cv2.resize(img,dsize=(m2,m2),interpolation=cv2.INTER_CUBIC)
                #print('sx',sx,'sy',sy,'m2',m2,'mid2',mid2,'img',img.shape)
                #print(r.ymin,r.ymax,r.xmin,r.xmax,':',mid2-sy//2,mid2+(sy-sy//2),mid2-sx//2,mid2+(sx-sx//2))
                dx = (m2-sx)//2
                dy = (m2-sy)//2
                #print('dx',dx,'dy',dy)
                d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = img[dy:dy+sy,dx:dx+sx]
            #d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = cv2.resize(img,dsize=(r.xmax-r.xmin,r.ymax-r.ymin),interpolation=cv2.INTER_LINEAR)
            #d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = cv2.resize(img,dsize=(r.xmax-r.xmin,r.ymax-r.ymin),interpolation=cv2.INTER_CUBIC)
            
            il.append(synlabel[fn[0:9]]-1)

        # for each rf, populate l[] with len(ir) [0,1] probabilities based on overlap between rf and ir rectangles
        if sc=='1x1':
            l2[i,il[0],0,0] = 1
        if sc=='2x2':
            l1[i,il[0],0,0] = 1
            l1[i,il[1],0,1] = 1
            l1[i,il[2],1,0] = 1
            l1[i,il[3],1,1] = 1
        if sc=='3x3':
            l0[i,il[0],0,0] = 1
            l0[i,il[1],0,1] = 1
            l0[i,il[2],0,2] = 1
            l0[i,il[3],1,0] = 1
            l0[i,il[4],1,1] = 1
            l0[i,il[5],1,2] = 1
            l0[i,il[6],2,0] = 1
            l0[i,il[7],2,1] = 1
            l0[i,il[8],2,2] = 1

#        if sc=='1x1':
#            for ll in range(1):
#                l[i,il[ll],ll] = 1.0
#        if sc=='2x2':
#            for ll in range(4):
#                l[i,il[ll],1+ll] = 1.0
#        if sc=='3x3':
#            for ll in range(9):
#                l[i,il[ll],1+4+ll] = 1.0

#        if sc=='1x1':
#            for rr in range(2):
#                for cc in range(2):
#                    for ll in range(1):
#                        l[i,rr,cc,il[ll]] = 1.0
#        if sc=='2x2':
#            for rr in range(2):
#                for cc in range(2):
#                    for ll in range(4):
#                        l[i,rr,cc,il[ll]] = 1.0
#        if sc=='3x3':
#            for rr in range(2):
#                for cc in range(2):
#                    for ll in range(9):
#                        l[i,rr,cc,il[ll]] = 1.0

#        if sc=='1x1':
#            l[i,0,0,il[0]] = 0.25
#            l[i,0,1,il[0]] = 0.25
#            l[i,1,0,il[0]] = 0.25
#            l[i,1,1,il[0]] = 0.25
#        if sc=='2x2':
#            l[i,0,0,il[0]] = 1.0
#            l[i,0,1,il[1]] = 1.0
#            l[i,1,0,il[2]] = 1.0
#            l[i,1,1,il[3]] = 1.0
#        if sc=='3x3':
#            l[i,0,0,il[0]] = 1.0
#            l[i,0,0,il[1]] = 0.5
#            l[i,0,0,il[3]] = 0.5
#            l[i,0,0,il[4]] = 0.25
#            l[i,0,1,il[2]] = 1.0
#            l[i,0,1,il[1]] = 0.5
#            l[i,0,1,il[5]] = 0.5
#            l[i,0,1,il[4]] = 0.25
#            l[i,1,0,il[6]] = 1.0
#            l[i,1,0,il[3]] = 0.5
#            l[i,1,0,il[7]] = 0.5
#            l[i,1,0,il[4]] = 0.25
#            l[i,1,1,il[8]] = 1.0
#            l[i,1,1,il[5]] = 0.5
#            l[i,1,1,il[7]] = 0.5
#            l[i,1,1,il[4]] = 0.25

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
    #l = np.rollaxis(l,-1,1)
    #return d,l
    l=[]
    l.append(l2[:,:,0,0])
    l.append(l1[:,:,0,0])
    l.append(l1[:,:,0,1])
    l.append(l1[:,:,1,0])
    l.append(l1[:,:,1,1])
    l.append(l0[:,:,0,0])
    l.append(l0[:,:,0,1])
    l.append(l0[:,:,0,2])
    l.append(l0[:,:,1,0])
    l.append(l0[:,:,1,1])
    l.append(l0[:,:,1,2])
    l.append(l0[:,:,2,0])
    l.append(l0[:,:,2,1])
    l.append(l0[:,:,2,2])
    l = np.stack(l,axis=-1)
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

model = Imagenet.Imagenet(encoder,alt=args.alt)
with open(args.log, 'a') as f:
    print(args,file=f)
    print(torchinfo.summary(model, input_size=(1, 3, args.roi, args.roi)),file=f)
if args.debug:
    torchinfo.summary(model, input_size=(1, 3, args.roi, args.roi))
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
    #print('y0',y0.shape)
    y0 = y0[:,:,0].reshape([-1,1000,1,1])

    x=torch.utils.data.default_convert(x0)
    x = x.to(device)
    y=torch.utils.data.default_convert(y0)
    #y = y.reshape([-1, 1000, 14])
    y = y.to(device)
     
    logits = model(x)
    #logits = logits.reshape([-1, 1000, 14])
    #print('logits',logits.shape,'y',y.shape)
    loss = criterion(logits,y)
    #loss *= 1+4+9
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
        torch.save(model, 'finetune.pt')

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
