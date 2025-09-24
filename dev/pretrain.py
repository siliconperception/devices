# Copyright (c) 2024 Silicon Perception Inc (www.siliconperception.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torchinfo ; print('torchinfo',torchinfo.__version__)
import argparse
import os
import queue
import threading
import subprocess
import datetime
import models
from datasets import load_dataset # HF

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eps', help='',default=1e-08, type=float)
parser.add_argument('--steps', help='number of gradient steps until exit',default=None, type=int)
parser.add_argument('--amsgrad', help='',default=False, action='store_true')
parser.add_argument('--weight_decay', help='',default=0.01, type=float)
parser.add_argument('--batch', help='batch size',default=50, type=int)
parser.add_argument('--learning_rate', help='',default=0.00001, type=float)
parser.add_argument('--alt', help='{repl,lite,proj}-{base,batchnorm}',default='lite-base')
parser.add_argument('--slow', help='gradient scaling for perception models',default=None, type=float) # 1/(81*81)
parser.add_argument('--beta', help='second adamw moment coefficient',default=0.999, type=float)
parser.add_argument('--freeze', help='freeze embed, lmhead, decoder layers',default=False, action='store_true')
parser.add_argument('--momentum', help='',default=0, type=float)
parser.add_argument('--nesterov', help='',default=False, action='store_true')
parser.add_argument('--save', help='steps between model checkpoints',default=1000, type=int)
parser.add_argument('--prompt', help='for periodic model generation during training',default='')
parser.add_argument('--bos', help='number of BOS steps',default=2, type=int)
parser.add_argument('--seqlen', help='',default=None, type=int)
parser.add_argument('--generate', help='sample model interval',default=100, type=int)
parser.add_argument('--monitor', help='number of gradient updates before logging',default=10, type=int)
parser.add_argument('--schedule', help='learning rate schedule',default='linear')
parser.add_argument('--start_factor', help='',default=1.0, type=float)
parser.add_argument('--end_factor', help='',default=1.0, type=float)
parser.add_argument('--period', help='learning rate parameter',default=1000, type=int)
parser.add_argument('--gamma', help='learning rate parameter',default=0.9, type=float)
parser.add_argument('--shuffle', help='',default=False, action='store_true')
parser.add_argument('--dataset', help='tiny, c4',default='c4')
parser.add_argument('--opt', help='pytorch optimizer {sgd, adamw}',default='adamw')
parser.add_argument('--epochs', help='number of training epochs',default=100, type=int)
parser.add_argument('--device', help='pytorch execution device',default=None)
parser.add_argument('--load', help='load pytorch state dict',default=None)
parser.add_argument('--checkpoint', help='checkpoint file name',default='checkpoint.pt')
parser.add_argument('--n_embd', help='',default=384, type=int)
parser.add_argument('--seed', help='random seed',default=None, type=int)
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--verbose', help='',default=False, action='store_true')
args = parser.parse_args()

# ------------
# log file
# ------------
if args.log is None:
    if not os.path.exists('log'):
        os.makedirs('log')
    args.date = subprocess.check_output(['/usr/bin/date', '+%Y.%m.%d-%H.%M.%S'])
    args.date = args.date.decode("utf-8")
    args.date = args.date.rstrip()
    args.log = 'log/log.{}'.format(args.date)
if args.device is None:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.seed is None:
    args.seed = int.from_bytes(os.urandom(4), byteorder="big")
    print('seed', args.seed)
print(args)
with open(args.log, 'a') as f:
    print('ARGS',args,file=f)
torch.manual_seed(args.seed)

# DATASET LOADER
if args.dataset=='tiny':
    dataset = load_dataset('roneneldan/TinyStories', streaming=True)
if args.dataset=='c4':
    dataset = load_dataset("allenai/c4", "en", streaming=True)
if args.shuffle:
    dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)
dataset = iter(dataset['train'])

sample_example=''
num_examples=0
BOS = b'\xFE'*args.bos

def worker(stop,q,dataset,args):
    global sample_example
    global num_examples
    e = args.batch*[b'']
    while not stop.is_set():
        for i in range(args.batch):
            while len(e[i]) < 2:
                example = next(dataset)
                example = example['text']
                example = example.encode(encoding='ASCII', errors='ignore')
                sample_example = example[0:100]
                if args.verbose:
                    print(example, len(example))
                    with open(args.log, 'a') as f:
                        print(example, len(example), file=f)
                e[i] += BOS+example
                num_examples +=1
        x=[]
        y=[]
        for i in range(args.batch):
            x.append(e[i][0])
            y.append(e[i][1])
            e[i] = e[i][1:]
        q.put((x,y))

stop = threading.Event()
stop.clear()
q = queue.Queue(maxsize=100) # training data generator
w = threading.Thread(target=worker, args=[stop,q,dataset,args], daemon=False)
w.start()

model = models.CNN_LM(args.n_embd, args.alt)

if args.load is not None:
    model.load_state_dict(torch.load(args.load, weights_only=True))

if args.freeze:
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.lmhead.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.embed.parameters():
        param.requires_grad = False

info = torchinfo.summary(model, col_names=["input_size","output_size","num_params"],
    input_data=[torch.zeros([1,args.n_embd,81,81]), torch.zeros([1,256,1,1])])
print(info)
with open(args.log, 'a') as f:
    print('TORCHINFO',info,file=f)

m = model.to(args.device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
if args.opt=='radam':
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.learning_rate, decoupled_weight_decay=True)
if args.opt=='adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9,args.beta), weight_decay=args.weight_decay, amsgrad=args.amsgrad, eps=args.eps)
if args.opt=='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.opt=='sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov)

if args.schedule=='cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.period)
elif args.schedule=='linear':
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.start_factor, end_factor=args.end_factor, total_iters=args.period)
elif args.schedule=='warmup':
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=args.start_factor, total_iters=args.period)

print(args)

if args.slow is not None:
    slow = []
    slow.extend(model.encoder.parameters())
    slow.extend(model.decoder.parameters())
    slow.extend(model.embed.parameters())
    slow.extend(model.lmhead.parameters())

larr=[]
garr=[]
ctx = torch.zeros([args.batch,args.n_embd,81,81])
ctx = ctx.to(args.device)
i=0
try:
    while True:
        if (i%args.save)==0:
            torch.save(model.state_dict(),args.checkpoint)
        if (i%args.generate)==0:
            model.eval()
            s,_ = model.generate(BOS+args.prompt.encode("utf-8"), 200)
            print('\n', s, '\n')
            with open(args.log, 'a') as f:
                print('\n', s, '\n', file=f)

        # predict next token, next context given current token, current context
        (x0,y0)=q.get()
        x = torch.zeros([args.batch,256,1,1])
        y = torch.zeros([args.batch,1,1], dtype=int)
        x[:,:,0,0] = F.one_hot(torch.tensor(x0),num_classes=256).float()
        y[:,0,0] = torch.tensor(y0)
        x, y = x.to(args.device), y.to(args.device)
        model.train()
        logits,_,loss = model(ctx, x, y)
        loss.backward()
        larr.append(loss.item())

        # Scale gradients for slow modules
        if args.slow is not None:
            with torch.no_grad():
                for param in slow:
                    if param.grad is not None:
                        param.grad *= args.slow
        optimizer.step()
        model.eval()
        _,nxt,_ = model(ctx, x, y) # new targets
    
        # monitor gradient
        total_norm = 0
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        garr.append(total_norm)
    
        optimizer.zero_grad()
        ctx = nxt.detach()
    
        if args.seqlen is not None and (i%args.seqlen)==0:
            ctx = torch.zeros([args.batch,args.n_embd,81,81])
            ctx = ctx.to(args.device)
    
        scheduler.step()
        if (i%args.monitor)==0:
            s = 'STEP i {:10} wall {} loss {:12.9f} grad {:12.6f} lr {:12.9f} mean {:12.6f} std {:12.6f} example {:10} {}'.format(
                i, datetime.datetime.now(), np.mean(larr[-args.monitor:]), np.mean(garr[-args.monitor:]), scheduler.get_last_lr()[0],
                torch.mean(ctx).item(), torch.std(ctx).item(), num_examples, sample_example[0:100])
            print(s)
            with open(args.log, 'a') as f:
                print(s,file=f)
        i+=1
        if args.steps is not None and i > args.steps:
            break

except KeyboardInterrupt:
    pass

print('\nSTOPPING THREADS')
stop.set()
while not q.empty(): # drain
    q.get()
w.join()
print('EXIT MAIN')

