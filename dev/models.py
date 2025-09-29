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

class CNN_ENCODER(nn.Module): # project one-hot token [V] to feature map [H,W,C]
    def __init__(self, n_embd, n_proj, alt):
        super().__init__()
        self.alt = alt
        if 'repl' in alt:
            self.layer1  = nn.Sequential(nn.Upsample(scale_factor=81, mode='nearest'))
        elif 'proj' in alt:
            self.layer1  = nn.Sequential(nn.Upsample(scale_factor=3, mode='nearest'))
            self.layer2  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Upsample(scale_factor=3, mode='nearest'))
            self.layer6  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer8  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer9  = nn.Sequential(nn.Upsample(scale_factor=3, mode='nearest'))
            self.layer10  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer11  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer12  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer13  = nn.Sequential(nn.Upsample(scale_factor=3, mode='nearest'))
            self.layer14  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer15  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer16  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer17  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=1, stride=1, padding=0))
        elif 'lite' in alt:
            layers = []
            for i in range(4):
                layers.append(nn.Upsample(scale_factor=3, mode='nearest'))
                layers.append(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1))
                layers.append(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1))
                layers.append(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1))
            layers.append(nn.Conv2d(n_proj, n_embd, kernel_size=1, stride=1, padding=0))
            self.layers = nn.ModuleList(layers)
    def forward(self, x):
        if 'repl' in self.alt:
            x = self.layer1(x)
        elif 'lite' in self.alt:
            for layer in self.layers:
                x = layer(x)
        elif 'proj' in self.alt:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
            x = self.layer8(x)
            x = self.layer9(x)
            x = self.layer10(x)
            x = self.layer11(x)
            x = self.layer12(x)
            x = self.layer13(x)
            x = self.layer14(x)
            x = self.layer15(x)
            x = self.layer16(x)
            x = self.layer17(x)
        return x

class CNN_DECODER(nn.Module): # project feature map [H,W,C] to token logits [V]
    def __init__(self, n_embd, n_proj, alt):
        super().__init__()
        self.alt = alt
        if 'dlite' in alt:
            self.layer1  = nn.Sequential(nn.Conv2d(n_embd, n_proj, kernel_size=3, stride=2, padding=0))
            self.layer2  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0))
            self.layer3  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0))
            self.layer4  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0))
            self.layer5  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0))
        elif 'base' in alt:
            self.layer1  = nn.Sequential(nn.Conv2d(n_embd, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer2  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer3  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0))
            self.layer4  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer5  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer6  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0))
            self.layer7  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer8  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer9  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0))
            self.layer10  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer11  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer12  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        if 'dlite' in self.alt:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
        elif 'base' in self.alt:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
            x = self.layer8(x)
            x = self.layer9(x)
            x = self.layer10(x)
            x = self.layer11(x)
            x = self.layer12(x)
        return x

class CNN_PROJECTOR(nn.Module): # project feature map [H,W,C] to [H,W,C]
    def __init__(self, n_embd, alt):
        super().__init__()
        self.alt = alt
        if 'jumbo' in alt:
            self.layer1  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer2  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer6  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer8  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer9  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer10  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer11  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer12  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer13  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer14  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer15  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer16  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer17  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer18  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            #self.layer19  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer19  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1))
        elif 'base' in alt:
            self.layer1  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer2  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer6  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer8  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer9  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer10  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer11  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer12  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=1, stride=1, padding=0), nn.ReLU())
        elif 'batchnorm' in alt:
            self.layer1  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer2  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer6  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer8  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer9  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer10  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer11  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer12  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        if 'jumbo' in self.alt:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
            x = self.layer8(x)
            x = self.layer9(x)
            x = self.layer10(x)
            x = self.layer11(x)
            x = self.layer12(x)
            x = self.layer13(x)
            x = self.layer14(x)
            x = self.layer15(x)
            x = self.layer16(x)
            x = self.layer17(x)
            x = self.layer18(x)
            x = self.layer19(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
            x = self.layer8(x)
            x = self.layer9(x)
            x = self.layer10(x)
            x = self.layer11(x)
            x = self.layer12(x)
        return x

class CNN_LM(nn.Module):
    def __init__(self, n_embd=384, n_proj=32, vocab=256, alt='jumbo-lite'):
        super().__init__()
        self.alt = alt
        self.n_embd = n_embd
        self.n_proj = n_proj
        self.vocab = vocab
        self.encoder = CNN_ENCODER(n_embd, n_proj, alt)
        self.decoder = CNN_DECODER(n_embd, n_proj, alt)
        self.projector = CNN_PROJECTOR(n_embd, alt)
        self.embed  = nn.Sequential(nn.Conv2d(vocab, n_proj, kernel_size=1, stride=1, padding=0))
        self.lmhead  = nn.Sequential(nn.Conv2d(n_proj, vocab, kernel_size=1, stride=1, padding=0))

    def forward(self, ctx, idx, targets=None): # idx and targets are both (B,T) tensor of integers
        tok = self.embed(idx)
        enc = self.encoder(tok)
        proj = self.projector(ctx)
        res = proj+enc
        dec = self.decoder(res)
        logits = self.lmhead(dec)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
        return logits,res,loss

    def generate(self, prompt, ntokens, ctx=None):
        ret=[]
        self.eval()
        device = next(self.parameters()).device
        x = torch.zeros([1,256,1,1])
        x = x.to(device)
        if ctx is None:
            ctx = torch.zeros([1,self.n_embd,81,81])
        ctx = ctx.to(device)

        for b in prompt:
            x[0,:,0,0] = F.one_hot(torch.tensor(b),num_classes=256).float()
            logits,nxt,_ = self.forward(ctx, x)
            ctx = nxt.detach()

        g=''
        for i in range(ntokens):
            try:
                probs = F.softmax(logits,dim=1)
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    g+='X'
                    continue
                probs = probs[0,:,0,0]
                idx = torch.multinomial(probs,num_samples=1)
                g+=chr(idx.item())
                x[0,:,0,0] = F.one_hot(idx,num_classes=256).float()
                logits,nxt,_ = self.forward(ctx, x)
                ctx = nxt.detach()
                f = ctx.cpu().numpy()
                f = np.squeeze(f)
                #f = np.mean(np.abs(f), axis=0) # L1 norm
                f = np.std(f, axis=0)
                ret.append(f)
            except:
                pass
        
        printable_chars = "".join(char for char in g if char=='\n' or char.isprintable())
        return printable_chars, np.array(ret)
