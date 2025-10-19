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
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import numpy as np
BOS = 50256

class CNN_ENCODER(nn.Module): # project token embedding [C] to feature map [H,W,C]
    def __init__(self, n_hidden, n_embd, n_proj, context, alt):
        super().__init__()
        self.alt = alt
        if 'repl' in alt:
            self.layer1  = nn.Sequential(nn.Upsample(scale_factor=context, mode='nearest'))
            self.layer2  = nn.Sequential(nn.Conv2d(n_embd, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer6  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Conv2d(n_proj, n_hidden, kernel_size=1, stride=1, padding=0))
        elif 'x27' in alt:
            layers = []
            for i in range(3):
                layers.append(nn.Upsample(scale_factor=3, mode='nearest'))
                layers.append(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1))
                layers.append(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1))
                layers.append(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1))
            layers.append(nn.Conv2d(n_proj, n_hidden, kernel_size=1, stride=1, padding=0))
            self.layers = nn.ModuleList(layers)
        elif 'x28' in alt:
            self.layer1  = nn.Sequential(nn.Upsample(scale_factor=7, mode='nearest'))
            self.layer2  = nn.Sequential(nn.Conv2d(n_embd, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))
            self.layer6  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer8  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer9  = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))
            self.layer10  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer11  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer12  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer13  = nn.Sequential(nn.Conv2d(n_proj, n_hidden, kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        if 'repl' in self.alt:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
        elif 'x27' in self.alt:
            for layer in self.layers:
                x = layer(x)
        elif 'x28' in self.alt:
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
        return x

class CNN_DECODER(nn.Module): # project feature map [H,W,C] to token logits [V]
    def __init__(self, n_hidden, n_embd, n_proj, context, alt):
        super().__init__()
        self.alt = alt
        if 'pool' in alt:
            self.layer1  = nn.Sequential(nn.Conv2d(n_hidden, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer2  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer6  = nn.Sequential(nn.Conv2d(n_proj, n_embd, kernel_size=1, stride=1, padding=0))
            self.layer7  = nn.AvgPool2d(context)
            #self.layer7  = nn.MaxPool2d(28)
        elif 'x27' in alt:
            self.layer1  = nn.Sequential(nn.Conv2d(n_hidden, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer2  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer3  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0))
            self.layer4  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer5  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer6  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0))
            self.layer7  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0))
            self.layer12  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=1, stride=1, padding=0))
        elif 'x28' in alt:
            self.layer1  = nn.Sequential(nn.Conv2d(n_hidden, n_proj, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer2  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer6  = nn.Sequential(nn.Conv2d(n_proj, n_proj, kernel_size=3, stride=2, padding=0), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Conv2d(n_proj, n_embd, kernel_size=3, stride=1, padding=0))
            #self.layer12  = nn.Sequential(nn.Conv2d(n_proj, n_embd, kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        if 'pool' in self.alt:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
        elif 'x27' in self.alt:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
            x = self.layer12(x)
        elif 'x28' in self.alt:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
            #x = self.layer12(x)
        return x

class CNN_PROJECTOR(nn.Module): # project feature map [H,W,C] to [H,W,C]
    def __init__(self, n_hidden, n_embd, alt):
        super().__init__()
        self.alt = alt
        if 'mini' in alt:
            #self.layer1  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding='same', dilation=2), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer2  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding='same', dilation=2), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer3  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding='same', dilation=2), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer4  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding='same', dilation=2), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer5  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding='same', dilation=2), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer6  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding='same', dilation=2), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer7  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding='same', dilation=2), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer8  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding='same', dilation=2), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer9  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding='same', dilation=2), nn.BatchNorm2d(n_hidden), nn.ReLU())
            self.layer1  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            self.layer2  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            self.layer6  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            self.layer8  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            self.layer9  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            self.layer10  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1))
        elif 'jumbo' in alt:
            #self.layer1  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=5, stride=1, padding=2), nn.ReLU())
            #self.layer2  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=5, stride=1, padding=2), nn.ReLU())
            #self.layer3  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=5, stride=1, padding=2), nn.ReLU())
            #self.layer4  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=5, stride=1, padding=2), nn.ReLU())
            #self.layer5  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=5, stride=1, padding=2), nn.ReLU())
            self.layer1  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer2  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer6  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer7  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer8  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer9  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer10  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer11  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer12  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer13  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer14  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer15  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer16  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer19  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=1, stride=1, padding=0))

            #self.layer1  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer2  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer3  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer4  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer5  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer6  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer7  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer8  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer9  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer10  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer11  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer12  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer13  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer14  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer15  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(n_hidden), nn.ReLU())
            #self.layer16  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU())
            #self.layer19  = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        if 'mini' in self.alt:
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
        elif 'jumbo' in self.alt:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            #x = self.layer6(x)
            #x = self.layer7(x)
            #x = self.layer8(x)
            #x = self.layer9(x)
            #x = self.layer10(x)
            #x = self.layer11(x)
            #x = self.layer12(x)
            #x = self.layer13(x)
            #x = self.layer14(x)
            #x = self.layer15(x)
            #x = self.layer16(x)
            x = self.layer19(x)
        return x

class CNN_LM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, n_hidden, n_embd, n_proj, context, vocab, alt):
        super().__init__()
        self.alt = alt
        self.n_hidden = n_hidden
        self.n_embd = n_embd
        self.n_proj = n_proj
        self.vocab = vocab
        self.projector = CNN_PROJECTOR(n_hidden, n_embd, alt)
        self.encoder = CNN_ENCODER(n_hidden, n_embd, n_proj, context, alt)
        self.decoder = CNN_DECODER(n_hidden, n_embd, n_proj, context, alt)

        if n_embd==768:
            self.tok_model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        elif n_embd==256:
            self.tok_model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-8M')
            self.tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories-8M')
        #print('tok_model', self.tok_model)
        #print('tokenizer', self.tokenizer)
        self.embed = self.tok_model.transformer.wte         # 50257->256
        self.lmhead = self.tok_model.lm_head    # 256->50257

        #self.embed  = nn.Sequential(nn.Conv2d(vocab, n_proj, kernel_size=1, stride=1, padding=0))
        #self.lmhead  = nn.Sequential(nn.Conv2d(n_proj, vocab, kernel_size=1, stride=1, padding=0))

    def forward(self, ctx, idx, targets=None): # idx and targets are both (B,T) tensor of integers
        #print('idx',idx.shape)
        tok = self.embed(idx)
        tok = tok.unsqueeze(-1)
        tok = tok.unsqueeze(-1)
        #print('tok',tok.shape)
        enc = self.encoder(tok)
        proj = self.projector(ctx)
        res = proj+enc
        dec = self.decoder(res)
        dec = torch.squeeze(dec, dim=(-2, -1))
        #print('dec',dec.shape)
        logits = self.lmhead(dec)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
        return logits,res,loss

    def generate(self, prompt='', ntokens=20, ctx=None):
        ret=[]
        tok=[]
        self.eval()
        device = next(self.parameters()).device
        #x = torch.zeros([1,1,1,1])
        #x = x.to(device)
        if ctx is None:
            #ctx = torch.zeros([1,self.n_hidden,81,81])
            #ctx = torch.zeros([1,self.n_hidden,27,27])
            #ctx = torch.zeros([1,self.n_hidden,28,28])
            ctx = torch.zeros([1,self.n_hidden,7,7])
        ctx = ctx.to(device)

        tok_prompt=[BOS]+self.tokenizer.encode(prompt)
        for idx in tok_prompt:
            #x[0,:,0,0] = F.one_hot(torch.tensor(b),num_classes=256).float()
            logits,nxt,_ = self.forward(ctx, torch.tensor(idx).to(device))
            #logits,nxt,_ = self.forward(ctx, x)
            ctx = nxt.detach()

        g=''
        for i in range(ntokens):
            #print('logits',logits.shape)
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                g+='X'
                continue
            #probs = probs[0,:,0,0]
            #print('probs',probs.shape,'idx',idx.shape)
            probs = probs[0]
            idx = torch.multinomial(probs,num_samples=1)
            t = self.tokenizer.decode(idx)
            tok.append(t)
            g += t
            #g+=chr(idx.item())
            #x[0,:,0,0] = F.one_hot(idx,num_classes=256).float()
            #logits,nxt,_ = self.forward(ctx, x)
            logits,nxt,_ = self.forward(ctx, idx)
            ctx = nxt.detach()
            f = ctx.cpu().numpy()
            f = np.squeeze(f)
            #f = np.mean(np.abs(f), axis=0) # L1 norm
            f = np.std(f, axis=0)
            ret.append(f)
        
        printable_chars = "".join(char for char in g if char=='\n' or char.isprintable())
        return printable_chars, np.array(ret), np.array(tok)
