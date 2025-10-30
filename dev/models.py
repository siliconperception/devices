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
#BOS = 50256

class CNN_ENCODER(nn.Module): # project token embedding [C] to feature map [H,W,C]
    def __init__(self, n_hidden, n_embd, n_enc, n_dec, context, vocab, alt):
        super().__init__()
        self.alt = alt
        if 'repl' in alt:
            layers = []
            layers.append(nn.Upsample(scale_factor=context, mode='nearest'))
            self.layers = nn.ModuleList(layers)
    def forward(self, x):
        if 'repl' in self.alt:
            for layer in self.layers:
                x = layer(x)
        return x

class CNN_DECODER(nn.Module): # project feature map [H,W,C] to token logits [V]
    def __init__(self, n_hidden, n_embd, n_enc, n_dec, context, vocab, alt):
        super().__init__()
        self.alt = alt
        if 'tree' in alt:
            layers = []
            layers.append(nn.Conv2d(n_enc+n_hidden, n_dec, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            d = np.ceil(np.log2(context)).astype(int)
            #print('d',d)
            for i in range(d):
                layers.append(nn.Conv2d(n_dec, n_dec, kernel_size=3, stride=2, padding=1))
                layers.append(nn.ReLU())
            layers.append(nn.Conv2d(n_dec, n_embd, kernel_size=1, stride=1, padding=0))
            self.layers = nn.ModuleList(layers)
        elif 'pool' in alt:
            layers = []
            layers.append(nn.Conv2d(n_enc+n_hidden, n_dec, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            for i in range(context-2-1):
                layers.append(nn.Conv2d(n_dec, n_dec, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU())
            layers.append(nn.Conv2d(n_dec, n_embd, kernel_size=1, stride=1, padding=0))
            layers.append(nn.AvgPool2d(context))
            #layers.append(nn.MaxPool2d(context))
            self.layers = nn.ModuleList(layers)
    def forward(self, x):
        if 'tree' in self.alt:
            for layer in self.layers:
                x = layer(x)
        elif 'pool' in self.alt:
            for layer in self.layers:
                x = layer(x)
        return x

class CNN_PROJECTOR(nn.Module): # project feature map [H,W,C] to [H,W,C]
    def __init__(self, n_hidden, n_embd, n_enc, n_dec, context, vocab, alt):
        super().__init__()
        self.alt = alt
        if 'mini' in alt:
            pass
        elif 'jumbo' in alt:
            layers = []
            layers.append(nn.Conv2d(n_enc+n_hidden, n_hidden, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            for i in range(context//2):
                layers.append(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU())
            layers.append(nn.Conv2d(n_hidden, n_hidden, kernel_size=1, stride=1, padding=0))
            self.layers = nn.ModuleList(layers)
    def forward(self, x):
        if 'mini' in self.alt:
            pass
        elif 'jumbo' in self.alt:
            for layer in self.layers:
                x = layer(x)
        return x

class CNN_LM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, n_hidden, n_embd, n_enc, n_dec, context, vocab, alt):
        super().__init__()
        self.alt = alt
        self.n_hidden = n_hidden
        self.n_embd = n_embd
        self.n_enc = n_enc
        self.n_dec = n_dec
        self.vocab = vocab
        self.context = context
        self.projector = CNN_PROJECTOR(n_hidden, n_embd, n_enc, n_dec, context, vocab, alt)
        self.encoder = CNN_ENCODER(n_hidden, n_embd, n_enc, n_dec, context, vocab, alt)
        self.decoder = CNN_DECODER(n_hidden, n_embd, n_enc, n_dec, context, vocab, alt)

        if n_embd==768:
            self.tok_model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        elif n_embd==256:
            self.tok_model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-8M')
            self.tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories-8M')
        self.embed = self.tok_model.transformer.wte         # 50257->256
        self.lmhead = self.tok_model.lm_head    # 256->50257

    def forward(self, ctx, idx, targets=None): # idx and targets are both (B,T) tensor of integers
        tok = self.embed(idx)
        tok = tok.unsqueeze(-1)
        tok = tok.unsqueeze(-1)
        enc = self.encoder(tok)
        proj = self.projector(ctx)
        res = torch.cat((enc, proj), dim=1)
        dec = self.decoder(res)
        dec = torch.squeeze(dec, dim=(-2, -1))
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
        if ctx is None:
            ctx = torch.zeros([1,self.n_enc+self.n_hidden,self.context,self.context])
        ctx = ctx.to(device)

        tok_prompt = self.tokenizer.encode('<|endoftext|>'+prompt)
        for idx in tok_prompt:
            logits,nxt,_ = self.forward(ctx, torch.tensor([idx]).to(device))
            ctx = nxt.detach()

        g=''
        for i in range(ntokens):
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                g+='X'
                continue
            probs = probs[0]
            idx = torch.multinomial(probs,num_samples=1)
            t = self.tokenizer.decode(idx)
            tok.append(t)
            g += t
            logits,nxt,_ = self.forward(ctx, idx)
            ctx = nxt.detach()
            f = ctx.cpu().numpy()
            f = np.squeeze(f)
            ret.append(np.std(f, axis=0))
        
        printable_chars = "".join(char for char in g if char=='\n' or char.isprintable())
        return printable_chars, np.array(ret), np.array(tok)
