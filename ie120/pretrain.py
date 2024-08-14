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
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--alt', help='encoder model alt type',default='alt1')
parser.add_argument('--encoder', help='encoder model name',default='ie120-050-240')
parser.add_argument('--save', help='output encoder model name',default=None)
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
parser.add_argument('--imagenet', help='imagenet dataset base directory',default='../../')
parser.add_argument('--scratch', help='start training from random weights',default=False, action='store_true')
parser.add_argument('--checkpoint', help='save timestamped checkpoint every 100000 batches',default=False, action='store_true')
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--train', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--batch', help='batch size',default=32, type=int)
parser.add_argument('--lr', help='initial learning rate',default=0.0005, type=float)
parser.add_argument('--show', help='display batches',default=False, action='store_true')
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

if args.scratch:
    encoder = ie120.IE120(args.alt)
    print('image encoder model initialized from scratch')
else:
    encoder = torch.load('{}'.format(args.encoder))
    print('image encoder model loaded')

encoder.train() # train mode

dataset = batch.Batch(args)

class Model(nn.Module):
    def __init__(self, encoder, alt='alt1'):
        super(Model, self).__init__()
        self.encoder = encoder
        self.alt = alt
        if alt=='alt1':
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

model = Model(encoder,alt=args.alt)
with open(args.log, 'a') as f:
    print(args,file=f)
    print(torchinfo.summary(model, input_size=(1, 3, args.roi, args.roi)),file=f)
if args.debug:
    torchinfo.summary(model, input_size=(1, 3, args.roi, args.roi))
device = torch.device('cuda')
model = model.to(device)

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
while i<1+args.train:
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
    mom = optimizer.param_groups[0]['momentum']
    s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} opt {:15} mom {:12.10f} nest {}'.format(
        i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,args.opt,mom,args.nesterov)
    print(s)
    with open(args.log, 'a') as f:
        print(s,file=f)
    
    if args.save is not None and (i%1000)==0:
        torch.save(encoder.state_dict(), '{}'.format(args.save))

    if args.show and (i%1)==0:
        img = x0[0]*255
        img = img.astype(np.uint8)
        img = np.swapaxes(img,0,-1)
        img = np.swapaxes(img,0,1)
        cv2.imshow('imagenet', img)
        cv2.waitKey(1)

#    # update momentum
#    if i==args.slow:
#        state = optimizer.state_dict()
#        state['param_groups'][0]['momentum'] = 0.90
#        optimizer.load_state_dict(state)

    # batch counter
    i+=1

print('STOPPING WORKERS')
stop.set()
for w in workers:
    while not q.empty():
        q.get()
    w.join()
print('EXIT MAIN')
