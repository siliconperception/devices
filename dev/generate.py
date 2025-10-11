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
import torchinfo
import argparse
import os
import models
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pretrained', help='load pretrained model from HF',default=False, action='store_true')
parser.add_argument('--delay', help='second between frames for --vis',default=0.1, type=float)
parser.add_argument('--cmap', help='color map for visualization',default='viridis')
parser.add_argument('--vis', help='visualize context',default=False, action='store_true')
parser.add_argument('--prompt', help='for periodic model generation during training',default='')
parser.add_argument('--bos', help='number of BOS steps',default=2, type=int)
parser.add_argument('--alt', help='CNN_LM variant',default='lite-base-jumbo')
parser.add_argument('--n', help='number of tokens to generate',default=200, type=int)
parser.add_argument('--load', help='load pytorch state dict',default=None)
parser.add_argument('--n_hidden', help='',default=256, type=int)
parser.add_argument('--n_embd', help='',default=256, type=int)
parser.add_argument('--n_proj', help='',default=32, type=int)
parser.add_argument('--vocab', help='',default=256, type=int)
parser.add_argument('--device', help='pytorch execution device',default=None)
parser.add_argument('--verbose', help='',default=False, action='store_true')
args = parser.parse_args()

if args.device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = args.device

if args.pretrained:
    model = models.CNN_LM(args.n_hidden, args.n_embd, args.n_proj, args.vocab, args.alt).from_pretrained('siliconperception/CNN_LM')
else:
    model = models.CNN_LM(args.n_hidden, args.n_embd, args.n_proj, args.vocab, args.alt)

m = model.to(device)

if args.load is not None:
    m.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))

#ctx = torch.zeros([1,args.n_hidden,81,81])
#ctx = torch.zeros([1,args.n_hidden,27,27])
ctx = torch.zeros([1,args.n_hidden,28,28])
ctx = ctx.to(device)
m.eval()
#BOS = b'\xFE'*args.bos
BOS = 50256

if args.vis:
    #print('prompt', BOS+args.prompt.encode("utf-8"))
    #s, mat = m.generate(BOS+args.prompt.encode("utf-8"), args.n, ctx)
    _,mat,tok = m.generate([[BOS]+m.tokenizer.encode(args.prompt)], args.n, ctx)
    print('-----------------------------------------------------------------------------------------')
    print('mat', mat.shape)
    print('tok', ''.join(tok))

    init=True
    s = 30*' '
    plt.ion()
    fig, ax = plt.subplots()
    for i,f in enumerate(mat):
        if init:
            img = ax.matshow(f, cmap=args.cmap) # Create the initial matshow object
            init=False
        else:
            img.set_data(f) # Update the data of the existing image

        #ax.set_title(f'Frame {i+1}')
        s += tok[i]
        ax.set_title('{:4d} : {}'.format(i,s[-30:].encode("utf-8")))
        plt.draw() # Redraw the figure
        plt.pause(0.1) # Pause for a short duration
        k = plt.waitforbuttonpress(timeout=args.delay)
        if k is not None:
            if k:
                while k:
                    k = not plt.waitforbuttonpress(timeout=0.01)
            else:
                exit()
        #print('k', k)

    plt.ioff() # Turn off interactive mode
    plt.show()
    exit()

s,_ = m.generate(BOS+args.prompt.encode("utf-8"), args.n, ctx)
print('-----------------------------------------------------------------------------------------')
print(s)
