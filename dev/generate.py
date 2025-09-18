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
parser.add_argument('--cmap', help='color map for visualization',default='gray')
parser.add_argument('--vis', help='visualize context',default=False, action='store_true')
parser.add_argument('--prompt', help='for periodic model generation during training',default='')
parser.add_argument('--bos', help='number of BOS steps',default=2, type=int)
parser.add_argument('--alt', help='CNN_LM variant',default='proj-base')
parser.add_argument('--n', help='number of tokens to generate',default=200, type=int)
parser.add_argument('--load', help='load pytorch state dict',default=None)
parser.add_argument('--n_embd', help='',default=384, type=int)
parser.add_argument('--device', help='pytorch execution device',default=None)
parser.add_argument('--verbose', help='',default=False, action='store_true')
args = parser.parse_args()

if args.device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = args.device

model = models.CNN_LM(args.n_embd, args.alt)
m = model.to(device)

if args.load is not None:
    m.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))

if args.verbose:
    torchinfo.summary(m, col_names=["input_size","output_size","num_params"], input_data=[torch.zeros([1,args.n_embd,81,81]), torch.zeros([1,256,1,1])])

ctx = torch.zeros([1,args.n_embd,81,81])
ctx = ctx.to(device)
m.eval()
BOS = b'\xFE'*args.bos

if args.vis:
    print('prompt', BOS+args.prompt.encode("utf-8"))
    s, mat = m.generate(BOS+args.prompt.encode("utf-8"), args.n, ctx)
    print('-----------------------------------------------------------------------------------------')
    print('mat', mat.shape)
    print('s', s)

    init=True
    s = 30*' ' + s
    plt.ion()
    fig, ax = plt.subplots()
    for i,f in enumerate(mat):
        if init:
            img = ax.matshow(f, cmap=args.cmap) # Create the initial matshow object
            init=False
        else:
            img.set_data(f) # Update the data of the existing image

        #ax.set_title(f'Frame {i+1}')
        ax.set_title('{:4d} : {}'.format(i,s[i:i+30].encode("utf-8")))
        plt.draw() # Redraw the figure
        plt.pause(0.1) # Pause for a short duration
        k = plt.waitforbuttonpress(timeout=0.1)
        if k is not None:
            if k:
                while k:
                    k = not plt.waitforbuttonpress(timeout=0.1)
            else:
                exit()
        #print('k', k)

    plt.ioff() # Turn off interactive mode
    plt.show()
    exit()

s,_ = m.generate(BOS+args.prompt.encode("utf-8"), args.n, ctx)
print('-----------------------------------------------------------------------------------------')
print(s)
