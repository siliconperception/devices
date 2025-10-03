import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torchinfo
import argparse
import os
import models

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--alt', help='CNN_LM variant',default='lite-base-jumbo')
parser.add_argument('--load', help='load pytorch state dict',default=None)
parser.add_argument('--n_embd', help='',default=384, type=int)
parser.add_argument('--n_proj', help='',default=32, type=int)
parser.add_argument('--vocab', help='',default=256, type=int)
#parser.add_argument('--device', help='pytorch execution device',default=None)
parser.add_argument('--device', help='pytorch execution device',default='cpu')
parser.add_argument('--push', help='',default=False, action='store_true')
parser.add_argument('--verbose', help='',default=False, action='store_true')
args = parser.parse_args()

if args.device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = args.device

model = models.CNN_LM(args.n_embd, args.n_proj, args.vocab, args.alt)
m = model.to(device)

if args.load is not None:
    m.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))

if args.verbose:
    torchinfo.summary(m, col_names=["input_size","output_size","num_params"], input_data=[torch.zeros([1,args.n_embd,27,27]), torch.zeros([1,args.vocab,1,1])])

if args.push:
    m.save_pretrained('CNN_LM')
    m.push_to_hub("siliconperception/CNN_LM")
