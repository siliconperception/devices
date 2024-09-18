# pretrain IE120 image encoder at 768x768 RGB resolution
import siliconperception ; print('siliconperception',siliconperception.__version__)
from siliconperception.IE120L import IE120L
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
import os
from collections import namedtuple
import scipy
import cv2

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--show', help='in batches',default=None, type=int)
parser.add_argument('--saveinterval', help='in batches',default=None, type=int)
parser.add_argument('--train', help='pretrain image encoder model',default=False, action='store_true')
parser.add_argument('--device', help='pytorch execution device',default='cpu')
parser.add_argument('--encoder', help='input encoder model name',default=None)
parser.add_argument('--save', help='output encoder model name',default='ie120l_pretrain.pt')
parser.add_argument('--avg', help='moving average window for lr and grad',default=100, type=int)
parser.add_argument('--factor', help='LR schedule param',default=0.1, type=float)
parser.add_argument('--slow', help='reduce momentum to 0.9 at this batch',default=-1, type=int)
parser.add_argument('--steps', help='LR scheduler steps',action='store', type=int,nargs='*')
parser.add_argument('--sched', help='LR scheduler type',default='linear')
parser.add_argument('--gamma', help='LR schedule param',default=1.0, type=float)
parser.add_argument('--nesterov', help='SGD param',default=False, action='store_true')
parser.add_argument('--momentum', help='SGD param',default=0.0, type=float)
parser.add_argument('--dampening', help='SGD param',default=0.0, type=float)
parser.add_argument('--opt', help='optimizer type',default='adamw')
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
#parser.add_argument('--scratch', help='start training from random weights',default=False, action='store_true')
#parser.add_argument('--checkpoint', help='save timestamped checkpoint every 100000 batches',default=False, action='store_true')
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--nbatch', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--batch', help='batch size',default=10, type=int)
parser.add_argument('--lr', help='initial learning rate',default=0.0005, type=float)
parser.add_argument('--seed', help='random seed',default=None, type=int)
parser.add_argument('--debug', help='verbose',default=False, action='store_true')
args = parser.parse_args()
args.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(args.seed)))
if args.log is None:
    if not os.path.exists('log'):
        os.makedirs('log')
    args.date = subprocess.check_output(['/usr/bin/date', '+%Y.%m.%d-%H.%M.%S'])
    args.date = args.date.decode("utf-8")
    args.date = args.date.rstrip()
    args.log = 'log/log.{}'.format(args.date)
print(args)
with open(args.log, 'a') as f:
    print('ARGS',args,file=f)

encoder = IE120L()
print('image encoder model initialized')

if args.encoder is not None:
    encoder.load_state_dict(torch.load('{}'.format(args.encoder)))
    print('image encoder model state_dict loaded')

class Batch:
    def __init__(self,args):
        self.synlabel={}
        self.labeltext={}
        meta = scipy.io.loadmat('{}/imagenet/devkit-1.0/data/meta.mat'.format(args.imagenet))
        synsets = meta['synsets']
        for i,s in enumerate(synsets):
            self.synlabel[s[0][1][0]] = s[0][0][0][0]
            self.labeltext[s[0][0][0][0]] = s[0][2][0]
        print('imagenet metadata loaded')
        
        with open('{}/imagenet/train/flist'.format(args.imagenet), 'r') as f:
            self.flist = f.readlines()
        print('imagenet flist loaded',len(self.flist))

    def generate_batch(self,args):
        d = np.zeros([args.batch,args.roi,args.roi,3]).astype(np.uint8)
        l0 = np.zeros([args.batch,1000,3,3]).astype(float) # class probabilities 3x3 feature map
        l1 = np.zeros([args.batch,1000,2,2]).astype(float) # class probabilities 2x2 feature map
        l2 = np.zeros([args.batch,1000,1,1]).astype(float) # class probabilities 1x1 feature map
    
        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
        choices = []
        if args.x1:
            choices.extend(1*['1x1'])
        if args.x2:
            choices.extend(1*['2x2'])
        if args.x3:
            choices.extend(1*['3x3'])
    
        for i in range(args.batch):
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
                    fn = random.choice(self.flist)
                    img = cv2.imread('{}/imagenet/train/{}'.format(args.imagenet,fn.rstrip()))
                    if img is not None:
                        break
                if args.centercrop:
                    side1 = min(img.shape[0],img.shape[1])
                    img = img[img.shape[0]//2-side1//2:img.shape[0]//2+side1//2,img.shape[1]//2-side1//2:img.shape[1]//2+side1//2]
                    sx = r.xmax-r.xmin
                    sy = r.ymax-r.ymin
                    m2 = max(sx,sy)
                    img = cv2.resize(img,dsize=(m2,m2),interpolation=cv2.INTER_CUBIC)
                    dx = (m2-sx)//2
                    dy = (m2-sy)//2
                    d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = img[dy:dy+sy,dx:dx+sx]
                il.append(self.synlabel[fn[0:9]]-1)
    
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
    
        d = d.astype(np.float32)/255.
        d = np.rollaxis(d,-1,1)
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

dataset = Batch(args)

class Pretrain(nn.Module):
    def __init__(self, encoder):
        super(Pretrain, self).__init__()
        self.encoder = encoder
        self.head0 = nn.Conv2d(512, 1000, kernel_size=(1,1), stride=1) # 3x3 feature map
        self.head1 = nn.Conv2d(512, 1000, kernel_size=(2,2), stride=1) # 2x2 feature map
        self.head2 = nn.Conv2d(512, 1000, kernel_size=(3,3), stride=1) # 1x1 feature map
    def forward(self, x):
        fmap = self.encoder(x)
        y0 = self.head0(fmap)
        y1 = self.head1(fmap)
        y2 = self.head2(fmap)
        y=[]
        y.append(y2[:,:,0,0])
        y.append(y1[:,:,0,0])
        y.append(y1[:,:,0,1])
        y.append(y1[:,:,1,0])
        y.append(y1[:,:,1,1])
        y.append(y0[:,:,0,0])
        y.append(y0[:,:,0,1])
        y.append(y0[:,:,0,2])
        y.append(y0[:,:,1,0])
        y.append(y0[:,:,1,1])
        y.append(y0[:,:,1,2])
        y.append(y0[:,:,2,0])
        y.append(y0[:,:,2,1])
        y.append(y0[:,:,2,2])
        y = torch.stack(y,dim=-1)
        return y

model = Pretrain(encoder)
with open(args.log, 'a') as f:
    print(args,file=f)
    print(torchinfo.summary(model, input_size=(1, 3, args.roi, args.roi)),file=f)
if args.debug:
    torchinfo.summary(model, input_size=(1, 3, args.roi, args.roi))

device = torch.device(args.device)
model = model.to(device)

if args.train:
    encoder.train() # train mode
    model.train() # train mode
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.opt=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.opt=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    if args.opt=='radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay,decoupled_weight_decay=True)
    if args.opt=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov,dampening=0.0)
    
    if args.sched=='linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=args.end_factor, total_iters=args.total_iters)
    if args.sched=='multi':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.gamma)
    if args.sched=='plat':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=args.factor,min_lr=0.0001)
    
    def worker(stop, q, args, dataset):
        while not stop.is_set():
            (d,l) = dataset.generate_batch(args)
            q.put((d,l))
    
    q = queue.Queue(maxsize=args.workers)
    stop = threading.Event()
    stop.clear()
    workers=[]
    for _ in range(args.workers):
        w = threading.Thread(target=worker, args=[stop,q,args,dataset], daemon=False)
        w.start()
        workers.append(w)
    
    i=1
    larr=[]
    garr=[]
    while i<1+args.nbatch:
        (x0,y0)=q.get()
    
        x=torch.utils.data.default_convert(x0)
        x = x.to(device)
        y=torch.utils.data.default_convert(y0)
        y = y.to(device)
         
        logits = model(x)
        loss = criterion(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        larr.append(loss.item())
        lavg = np.mean(larr[-args.avg:])
        if ((i%args.step)==0) or args.sched=='multi':
            scheduler.step(lavg)
    
        # compute gradient
        total_norm = 0
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
    
        # print batch statistics
        garr.append(total_norm)
        gavg = np.mean(garr[-args.avg:])
        lr = optimizer.param_groups[0]['lr']
        #mom = optimizer.param_groups[0]['momentum']
        s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} opt {:15} mom {:12.10f} nest {}'.format(
            i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,args.opt,args.momentum,args.nesterov)
        print(s)
        with open(args.log, 'a') as f:
            print(s,file=f)
        
        if args.saveinterval is not None and (i%args.saveinterval)==0:
            torch.save(encoder.state_dict(), '{}'.format(args.save))
            encoder.save_pretrained('ie120l_{}'.format(siliconperception.__version__))
            encoder.push_to_hub("siliconperception/IE120L")
    
        if args.show is not None and (i%args.show)==0:
            img = x0[0]*255
            img = img.astype(np.uint8)
            img = np.swapaxes(img,0,-1)
            img = np.swapaxes(img,0,1)
            cv2.imshow('imagenet', img)
            cv2.waitKey(1)
    
        # batch counter
        i+=1
    
    print('STOPPING WORKERS')
    stop.set()
    for w in workers:
        while not q.empty():
            q.get()
        w.join()
    print('EXIT MAIN')
    exit()
