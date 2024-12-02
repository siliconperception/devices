# train and test imagenet classification task using pretrained image encoder
import siliconperception ; print('siliconperception',siliconperception.__version__)
from siliconperception.IE120NX import IE120NX
from siliconperception.IE120L import IE120L
import numpy as np
import torch
import torch.nn as nn
import torchinfo
import cv2
import argparse
from pprint import pprint
import random
import threading
import queue
import subprocess
import os
import datetime
from collections import namedtuple
import scipy
import cv2
from transformers import AutoModel
import timm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--alt', help='encoder model alt type {convnext,ie120nx,ie120l}',default='convnext')
#parser.add_argument('--convnext', help='use convnext reference encoder',default=False, action='store_true')
parser.add_argument('--imagenet', help='imagenet dataset base directory',default='../')
#parser.add_argument('--data', help='train or val',default='train')
parser.add_argument('--device', help='pytorch execution device',default='cuda')
#parser.add_argument('--null', help='null probability threshold',default=0.5, type=float)
parser.add_argument('--decoder', help='decoder model name',default=None)
parser.add_argument('--nbatch', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--save', help='trained model name',default='imagenet.pt')
parser.add_argument('--lr', help='initial learning rate',default=0.1, type=float)
parser.add_argument('--weight_decay', help='L2 penalty',default=0.0, type=float)
parser.add_argument('--nesterov', help='SGD param',default=False, action='store_true')
parser.add_argument('--momentum', help='SGD param',default=0.0, type=float)
parser.add_argument('--factor', help='LR schedule param',default=0.7, type=float)
parser.add_argument('--step', help='LR scheduler batches per step',default=1000000000, type=int)
parser.add_argument('--avg', help='moving average window for lr and grad',default=100, type=int)
parser.add_argument('--workers', help='number of threads for batch generation',default=20, type=int)
parser.add_argument('--opt', help='optimizer type',default='sgd')
parser.add_argument('--sched', help='LR scheduler type',default='linear')
parser.add_argument('--end_factor', help='LR linear schedule parameter',default=1./50, type=float)
parser.add_argument('--total_iters', help='LR linear schedule parameter',default=10, type=float)
parser.add_argument('--batch', help='batch size',default=1, type=int)
parser.add_argument('--train', help='train encoder-decoder model',default=False, action='store_true')
parser.add_argument('--test', help='test encoder-decoder model',default=False, action='store_true')
parser.add_argument('--ann', help='annotation json file',default='../coco/annotations/instances_train2017.json')
parser.add_argument('--img', help='image directory',default='../coco/train2017')
parser.add_argument('--encoder', help='encoder model name',default=None)
parser.add_argument('--freeze', help='freeze encoder weights',default=False, action='store_true')
parser.add_argument('--roi', help='input image x/y size',default=768, type=int)
parser.add_argument('--debug', help='verbose',default=False, action='store_true')
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--show', help='display batches',default=False, action='store_true')
args = parser.parse_args()
if args.log is None:
    args.date = subprocess.check_output(['/usr/bin/date', '+%Y.%m.%d-%H.%M.%S'])
    args.date = args.date.decode("utf-8")
    args.date = args.date.rstrip()
    if not os.path.exists('log'):
        os.makedirs('log')
    args.log = 'log/log.{}'.format(args.date)
print(args)
with open(args.log, 'a') as f:
    print('ARGS',args,file=f)

#mamba = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
#for param in mamba.parameters():
#    param.requires_grad = False
#mamba.eval() # eval mode
#print('mamba loaded,frozen',self.mamba.config.mean,self.mamba.config.std)

if args.alt=='convnext':
    args.roi=224
    encoder = timm.create_model( 'convnextv2_tiny.fcmae', pretrained=True, features_only=True,)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder=encoder.eval()
    data_config = timm.data.resolve_model_data_config(encoder)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    args.dmean = transforms.transforms[-1].mean.numpy()
    args.dstd = transforms.transforms[-1].std.numpy()
    s=torchinfo.summary(encoder,col_names=["input_size","output_size","num_params"],input_size=(1,3,224,224))
    print(s)
    print('convnextv2 reference encoder loaded and frozen')
if args.alt=='ie120nx':
    encoder = IE120NX.from_pretrained('siliconperception/IE120NX')
    print('IE120NX pretrained image encoder model loaded')
if args.alt=='ie120l':
    encoder = IE120L.from_pretrained('siliconperception/IE120L')
    print('IE120L pretrained image encoder model loaded')
    
if args.freeze:
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval() # eval mode
    print('image encoder model frozen')
else:
    encoder.train() # train mode

class Decoder(nn.Module):
    def __init__(self, encoder, alt=None):
        super(Decoder, self).__init__()
        self.encoder = encoder
        self.alt = alt
        if self.alt=='convnext' or self.alt=='ie120nx':
            self.layer1  = nn.Conv2d(768, 100, kernel_size=1, stride=1) # linearly project each feature
            self.layerp = nn.Conv2d(100, 1000, kernel_size=7, stride=1) # linear projection to 1000 imagenet classes
        if self.alt=='ie120l':
            self.layerp = nn.Conv2d(512, 1000, kernel_size=3, stride=1) # linear projection to 1000 imagenet classes

    def forward(self, x):
        if self.alt=='convnext':
            fmap = self.encoder(x)[3]
            y = self.layer1(fmap)
            y = self.layerp(y)
            return y[:,:,0,0]
        if self.alt=='ie120nx':
            fmap = self.encoder(x)
            y = self.layer1(fmap)
            y = self.layerp(y)
            return y[:,:,0,0]
        if self.alt=='ie120l':
            fmap = self.encoder(x)
            y = self.layerp(fmap)
            return y[:,:,0,0]

model = Decoder(encoder,alt=args.alt)
torchinfo.summary(model, input_size=(1, 3, args.roi, args.roi))
device = torch.device(args.device)
model = model.to(device)

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
            self.train_flist = f.readlines()
        print('imagenet flist loaded - TRAINING DISTRIBUTION',len(self.train_flist))
        with open('{}/imagenet/val/flist'.format(args.imagenet), 'r') as f:
            self.val_flist = f.readlines()
        print('imagenet flist loaded - VALIDATION DISTRIBUTION',len(self.val_flist))
        with open('{}/imagenet/devkit-1.0/data/ILSVRC2010_validation_ground_truth.txt'.format(args.imagenet), 'r') as f:
            self.val_label = [int(line) for line in f.readlines()]
        print('imagenet labels loaded - VALIDATION DISTRIBUTION',len(self.val_flist))
        #for i,foo in enumerate(self.val_flist):
        #    print('i',i,'foo',foo)
        #    print(int(foo[15:15+8]),self.val_label[int(foo[15:15+8])]-1)

    def generate_batch(self,args,train=True):
        d = np.zeros([args.batch,args.roi,args.roi,3]).astype(np.uint8)
        l = np.zeros([args.batch]).astype(int) # class labels
        for i in range(args.batch):
            while True:
                if train:
                    fn = random.choice(self.train_flist)
                    img = cv2.imread('{}/imagenet/train/{}'.format(args.imagenet,fn.rstrip()))
                else:
                    fn = random.choice(self.val_flist)
                    img = cv2.imread('{}/imagenet/val/{}'.format(args.imagenet,fn.rstrip()))
                if img is not None:
                    break
            side1 = min(img.shape[0],img.shape[1])
            img = img[img.shape[0]//2-side1//2:img.shape[0]//2+side1//2,img.shape[1]//2-side1//2:img.shape[1]//2+side1//2]
            img = cv2.resize(img,dsize=(args.roi,args.roi),interpolation=cv2.INTER_CUBIC)
            d[i] = img
            if train:
                l[i] = self.synlabel[fn[0:9]]-1
            else:
                l[i] = self.val_label[int(fn[15:15+8])-1]-1

        d = d/255.
        if args.alt=='convnext':
            d = np.subtract(d,args.dmean)
            d = np.divide(d,args.dstd)

        #d = np.subtract(d,self.mamba.config.mean)
        #d = np.divide(d,self.mamba.config.std)
        d = np.rollaxis(d,-1,1)
        d = d.astype(np.float32)
        return d,l

if args.train:
    dataset = Batch(args)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() # we will use class indices for softmax loss
    if args.opt=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.opt=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    if args.opt=='radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay,decoupled_weight_decay=True)
    if args.opt=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov,dampening=0.0)
    if args.sched=='plat':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=args.factor,min_lr=0.0001)
    if args.sched=='linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=args.end_factor, total_iters=args.total_iters)
    
    def worker(stop, q, args, dataset,train):
        while not stop.is_set():
            (d,l) = dataset.generate_batch(args,train=train)
            q.put((d,l))
    
    q = queue.Queue(maxsize=args.workers)
    stop = threading.Event()
    stop.clear()
    workers=[]
    for _ in range(args.workers):
        w = threading.Thread(target=worker, args=[stop,q,args,dataset,True], daemon=False)
        w.start()
        workers.append(w)
    
    qtest = queue.Queue(maxsize=100)
    testers=[]
    for _ in range(args.workers//10):
        w = threading.Thread(target=worker, args=[stop,qtest,args,dataset,False], daemon=False)
        w.start()
        testers.append(w)
    
    i=1
    larr=[] # loss
    garr=[] # gradient
    aarr=[0] # accuracy
    while i<1+args.nbatch:
        (x0,y0)=q.get()
        #print('x0',np.amin(x0),np.amax(x0))
        #print('y0',np.amin(y0),np.amax(y0))
        x=torch.utils.data.default_convert(x0)
        x = x.to(device)
        y=torch.utils.data.default_convert(y0)
        y = y.to(device)
         
        model.train()
        logits = model(x)
        loss = criterion(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        larr.append(loss.item())
        lavg = np.mean(larr[-args.avg:])
        if args.sched is not None and (i%args.step)==0:
            if args.sched=='plat':
                scheduler.step(lavg)
            else:
                scheduler.step()
    
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
        try:
            mom = optimizer.param_groups[0]['momentum']
        except:
            mom=0
        aavg = np.mean(aarr[-args.avg:])
        #s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} accuracy {:12.6f} opt {:15} mom {:12.10f} nest {}'.format(
        #    i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,aavg,args.opt,mom,args.nesterov)
        s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} opt {:15} accuracy {:12.6f}'.format(
            i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,args.opt+'_'+args.sched,aavg)
        print(s)
        with open(args.log, 'a') as f:
            print(s,file=f)
        
        #if args.save is not None and (i%1000)==0:
        #    torch.save(model, '{}'.format(args.save))
    
        if args.test and (i%1000)==0:
            model.eval()
            for j in range(100):
                (d,l) = qtest.get() # sample a batch from the validation distribution
                x=torch.utils.data.default_convert(d)
                x = x.to(device)
                y=torch.utils.data.default_convert(l)
                y = y.to(device)
                logits = model(x)
                prob = torch.nn.functional.softmax(logits,dim=-1)
                prob = prob.cpu().detach().numpy()
                pred = np.argmax(prob,axis=-1)
                acc = np.mean(pred==l)
                if (j%10)==0:
                    print('TEST',j,acc)
                    #print('prob',prob,'pred',pred,'l',l)
                aarr.append(acc)

        # batch counter
        i+=1
    
    print('STOPPING WORKERS')
    stop.set()
    for w in workers:
        while not q.empty():
            q.get()
        w.join()
    for w in testers:
        while not qtest.empty():
            qtest.get()
        w.join()
    print('EXIT MAIN')
    exit()
