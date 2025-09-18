import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class CNN_ENCODER(nn.Module): # project token [C] to feature map [H,W,C]
    def __init__(self, n_embd, alt):
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
            self.layer1  = nn.Sequential(nn.Upsample(scale_factor=3, mode='nearest'))
            self.layer2  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Upsample(scale_factor=3, mode='nearest'))
            self.layer4  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Upsample(scale_factor=3, mode='nearest'))
            self.layer6  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Upsample(scale_factor=3, mode='nearest'))
            self.layer8  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.layer9  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=1, stride=1, padding=0), nn.ReLU())
    def forward(self, x):
        if 'repl' in self.alt:
            x = self.layer1(x)
        elif 'lite' in self.alt:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
            x = self.layer8(x)
            x = self.layer9(x)
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

class CNN_DECODER(nn.Module): # project feature map [H,W,C] to token [C]
    def __init__(self, n_embd, alt):
        super().__init__()
        if 'base' in alt:
            self.layer1  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer2  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=2, padding=0), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer6  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=2, padding=0), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer8  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer9  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=2, padding=0), nn.ReLU())
            self.layer10  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer11  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.ReLU())
            self.layer12  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=1, stride=1, padding=0), nn.ReLU())
        elif 'batchnorm' in alt:
            self.layer1  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer2  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer3  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer4  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer5  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer6  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer7  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer8  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer9  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer10  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer11  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
            self.layer12  = nn.Sequential(nn.Conv2d(n_embd, n_embd, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(n_embd), nn.ReLU())
    def forward(self, x):
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
        if 'base' in alt:
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
        if 'batchnorm' in alt:
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
    def __init__(self, n_embd, alt='proj-base'):
        super().__init__()
        self.n_embd = n_embd
        self.alt = alt

        self.encoder = CNN_ENCODER(n_embd, alt)
        self.decoder = CNN_DECODER(n_embd, alt)
        self.projector = CNN_PROJECTOR(n_embd, alt)
        self.embed  = nn.Sequential(nn.Conv2d(256, n_embd, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.lmhead  = nn.Sequential(nn.Conv2d(n_embd, 256, kernel_size=1, stride=1, padding=0))

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

        #prompt = prompt.encode(encoding='ASCII', errors='ignore')
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
        #print(g)
        printable_chars = "".join(char for char in g if char=='\n' or char.isprintable())
        #print(printable_chars)
        return printable_chars, np.array(ret)
