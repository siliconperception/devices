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

import torch ; print('torch', torch.__version__)
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
parser.add_argument('--context', help='size of context feature map',default=28, type=int)
parser.add_argument('--slow_acc', help='slow model gradient accumulation',default=1, type=int)
parser.add_argument('--superfreeze', help='freeze embed, lmhead, encoder, decoder layers',default=False, action='store_true')
parser.add_argument('--warmup', help='LR schedule param for warmcool',default=4000, type=int)
parser.add_argument('--hold', help='LR schedule param for warmcool',default=40000, type=int)
parser.add_argument('--warmdown', help='LR schedule param for warmcool',default=40000, type=int)
parser.add_argument('--eps', help='',default=1e-08, type=float)
parser.add_argument('--steps', help='number of gradient steps until exit',default=None, type=int)
parser.add_argument('--amsgrad', help='',default=False, action='store_true')
parser.add_argument('--weight_decay', help='WARNING MUST BE ZERO FOR CNN ???',default=0.0, type=float)
parser.add_argument('--batch', help='batch size',default=50, type=int)
parser.add_argument('--learning_rate', help='',default=0.00001, type=float)
#parser.add_argument('--slow_lr', help='',default=0.00001, type=float)
parser.add_argument('--alt', help='{repl,lite,proj}-{base,batchnorm}',default='free-jumbo')
parser.add_argument('--beta', help='second adamw moment coefficient',default=0.999, type=float)
parser.add_argument('--freeze', help='freeze embed, lmhead layers',default=False, action='store_true')
parser.add_argument('--momentum', help='',default=0, type=float)
parser.add_argument('--nesterov', help='',default=False, action='store_true')
parser.add_argument('--prompt', help='for periodic model generation during training',default='')
parser.add_argument('--generate', help='sample model interval',default=100, type=int)
parser.add_argument('--monitor', help='number of gradient updates before logging',default=10, type=int)
parser.add_argument('--schedule', help='learning rate schedule',default='linear')
parser.add_argument('--start_factor', help='',default=1.0, type=float)
parser.add_argument('--end_factor', help='',default=1.0, type=float)
parser.add_argument('--period', help='learning rate parameter',default=1000, type=int)
#parser.add_argument('--gamma', help='learning rate parameter',default=0.9, type=float)
parser.add_argument('--shuffle', help='',default=False, action='store_true')
parser.add_argument('--dataset', help='tiny, c4',default='c4')
parser.add_argument('--opt', help='pytorch optimizer {sgd, adamw}',default='adamw')
#parser.add_argument('--epochs', help='number of training epochs',default=100, type=int)
parser.add_argument('--device', help='pytorch execution device',default=None)
parser.add_argument('--load', help='load pytorch state dict',default=None)
parser.add_argument('--checkpoint', help='steps between model checkpoints',default=1000, type=int)
parser.add_argument('--save', help='checkpoint file name',default='checkpoint.pt')
parser.add_argument('--n_hidden', help='',default=192, type=int)
parser.add_argument('--n_embd', help='',default=256, type=int)
parser.add_argument('--n_proj', help='',default=64, type=int)
parser.add_argument('--vocab', help='',default=256, type=int)
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

BOS = 50256
model = models.CNN_LM(args.n_hidden, args.n_embd, args.n_proj, args.context, args.vocab, args.alt)
print('vocab_size', model.tokenizer.vocab_size)

sample_example=''
num_examples=0

def worker(stop,q,dataset,args):
    global sample_example
    global num_examples
    #e = args.batch*[b'']
    e = [[] for _ in range(args.batch)]
    while not stop.is_set():
        for i in range(args.batch):
            while len(e[i]) < 2:
                example = next(dataset)
                example = example['text']
                sample_example = example.encode(encoding='ASCII', errors='ignore')
                sample_example = sample_example[0:100]
                example = model.tokenizer.encode(example)
                e[i] += [BOS]+example
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
q = queue.Queue(maxsize=args.batch) # training data generator
w = threading.Thread(target=worker, args=[stop,q,dataset,args], daemon=False)
w.start()

if args.load is not None:
    model.load_state_dict(torch.load(args.load, weights_only=True))

if args.superfreeze:
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.embed.parameters():
        param.requires_grad = False
    for param in model.lmhead.parameters():
        param.requires_grad = False

if args.freeze:
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.embed.parameters():
        param.requires_grad = False
    for param in model.lmhead.parameters():
        param.requires_grad = False

info = torchinfo.summary(model, col_names=["input_size","output_size","num_params"],
    input_data=[torch.zeros([1,args.n_hidden,args.context,args.context]), torch.zeros([1],dtype=torch.int32)])
print(info)
with open(args.log, 'a') as f:
    print('TORCHINFO',info,file=f)

model = model.to(args.device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

if args.opt=='adamw':
    opt_proj = torch.optim.AdamW([
        {'params': model.projector.parameters(), 'lr': args.learning_rate},
        ], lr=args.learning_rate, betas=(0.9,args.beta), weight_decay=args.weight_decay, amsgrad=args.amsgrad, eps=args.eps)
    opt_slow = torch.optim.AdamW([
        {'params': model.decoder.parameters(), 'lr': args.learning_rate},
        {'params': model.encoder.parameters(), 'lr': args.learning_rate},
        {'params': model.lmhead.parameters(), 'lr': args.learning_rate}
        ], lr=args.learning_rate, betas=(0.9,args.beta), weight_decay=args.weight_decay, amsgrad=args.amsgrad, eps=args.eps)
elif args.opt=='sgd':
    opt_proj = torch.optim.SGD([
        {'params': model.projector.parameters(), 'lr': args.learning_rate},
        ], lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov) # defaults
    opt_slow = torch.optim.SGD([
        {'params': model.decoder.parameters(), 'lr': args.learning_rate},
        {'params': model.encoder.parameters(), 'lr': args.learning_rate},
        {'params': model.lmhead.parameters(), 'lr': args.learning_rate}
        ], lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay) # defaults

if args.schedule=='whc':
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.start_factor, end_factor=1.0, total_iters=args.warmup)
    hold = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=args.hold)
    cool = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=args.end_factor, total_iters=args.warmdown)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm, hold, cool], milestones=[args.warmup,args.hold])
elif args.schedule=='cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.period)
elif args.schedule=='linear':
    #sched_proj = torch.optim.lr_scheduler.LinearLR(opt_proj, start_factor=1, end_factor=1, total_iters=0)
    sched_proj = torch.optim.lr_scheduler.LinearLR(opt_proj, start_factor=args.start_factor, end_factor=args.end_factor, total_iters=args.period)
    sched_slow = torch.optim.lr_scheduler.LinearLR(opt_slow, start_factor=args.start_factor, end_factor=args.end_factor, total_iters=args.period)
elif args.schedule=='warmup':
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=args.start_factor, total_iters=args.period)

print(args)

larr=[]
garr=[]
ctx = torch.zeros([args.batch,args.n_hidden,args.context,args.context])
ctx = ctx.to(args.device)
i=0
try:
    while True:
        if (i%args.checkpoint)==0:
            torch.save(model.state_dict(),args.save)

        # periodically log a sample from the model
        if (i%args.generate)==0:
            model.eval()
            s,_,_ = model.generate(args.prompt, 50)
            print('\n', s, '\n')
            with open(args.log, 'a') as f:
                print('\n', s, '\n', file=f)

        # predict (next token, next context) given (current token, current context)
        (x0,y0)=q.get()
        x = torch.tensor(x0).to(args.device)
        y = torch.tensor(y0).to(args.device)
        model.train()
        logits,_,loss = model(ctx, x, y)
        loss.backward()
        #optimizer.step()
        opt_proj.step()
        if (i%args.slow_acc)==0:
            opt_slow.step()
        model.eval()
        _,nxt,_ = model(ctx, x, y) # compute context using updated model
        ctx = nxt.detach() # state machine
    
        # monitor loss and gradient
        larr.append(loss.item())
        total_norm = 0
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        garr.append(total_norm)
        if (i%args.monitor)==0:
            s = 'STEP i {:10} wall {} loss {:12.9f} grad {:12.6f} lr {:10.9f} mean {:12.6f} std {:12.6f} example {:10} {}'.format(
                i, datetime.datetime.now(), np.mean(larr[-args.monitor:]), np.mean(garr[-args.monitor:]), sched_proj.get_last_lr()[0],
                torch.mean(ctx).item(), torch.std(ctx).item(), num_examples, sample_example[0:100])
            print(s)
            with open(args.log, 'a') as f:
                print(s,file=f)

        # LR schedulers
        sched_proj.step()
        sched_slow.step()

        opt_proj.zero_grad()
        if (i%args.slow_acc)==0:
            opt_slow.zero_grad()
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

