import siliconperception ; print('siliconperception',siliconperception.__version__)
import torch
import torchinfo ; print('torchinfo',torchinfo.__version__)
import numpy as np
import timm
import argparse
import cv2
import scipy
import numpy as np
import random
import datetime
import queue
import threading
import subprocess
import os
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from collections import OrderedDict

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eta_min', help='LR cosint schedule parameter',default=1., type=float)
parser.add_argument('--tmax', help='LR cosint schedule parameter',default=1., type=float)
parser.add_argument('--mode', help='{finetune_resnet18,finetune_encoder,pretrain}',default='finetune_resnet18')
parser.add_argument('--verbose', help='logging',default=False, action='store_true')
parser.add_argument('--push', help='push encoder model to HF',default=False, action='store_true')
parser.add_argument('--saveinterval', help='push to HF every saveinterval batches',default=100000, type=int)
parser.add_argument('--saveoffset', help='push to HF every saveinterval batches',default=10000, type=int)
#parser.add_argument('--slide', help='slide training data distribution if gradient greater',default=-1, type=float)
#parser.add_argument('--cycle', help='cycle through samples',default=False, action='store_true')
#parser.add_argument('--shuffle', help='shuffle flist every args.shuffle batches',default=-1, type=int)
#parser.add_argument('--nsamples', help='size of flist in samples',default=-1, type=int)
#parser.add_argument('--lr2', help='learning rate distribution parameter',default=0.1, type=float)
#parser.add_argument('--warmup', help='number of batches at initial LR',default=0, type=int)
#parser.add_argument('--stub', help='generate stub for perceptron',default=False, action='store_true')
#parser.add_argument('--nf_llim', help='loss threshold for flist bump',default=0.01, type=float)
#parser.add_argument('--min_lr', help='minimum learning rate',default=0.00001, type=float)
#parser.add_argument('--patience', help='LR param',default=10, type=int)
parser.add_argument('--loss', help='loss function',default='hyb')
#parser.add_argument('--prec', help='scaling factor for reference model output in sigma',default=2, type=float)
parser.add_argument('--show', help='display batches',default=False, action='store_true')
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--nbatch', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--lr', help='initial learning rate',default=0.00001, type=float)
parser.add_argument('--device', help='pytorch execution device',default='cuda')
#parser.add_argument('--pretrain', help='pretrain image encoder model',default=False, action='store_true')
#parser.add_argument('--finetune', help='test pretrained image encoder model',default=False, action='store_true')
parser.add_argument('--encoder', help='input encoder model name',default=None)
parser.add_argument('--alt', help='encoder model alt type',default='large')
parser.add_argument('--imagenet', help='imagenet dataset base directory',default='../')
parser.add_argument('--batch', help='batch size',default=10, type=int)
#parser.add_argument('--roi', help='input image x/y size',default=896, type=int)
#parser.add_argument('--x1', help='include 1x1 examples',default=False, action='store_true')
#parser.add_argument('--x2', help='include 2x2 examples',default=False, action='store_true')
#parser.add_argument('--x3', help='include 3x3 examples',default=False, action='store_true')
#parser.add_argument('--resize', help='resize scale',default=0.0, type=float)
parser.add_argument('--workers', help='number of threads for batch generation',default=5, type=int)
parser.add_argument('--save', help='output encoder model name',default=None)
parser.add_argument('--avg', help='moving average window for lr and grad',default=100, type=int)
#parser.add_argument('--factor', help='LR schedule param',default=0.1, type=float)
#parser.add_argument('--steps', help='LR scheduler steps',action='store', type=int,nargs='*')
parser.add_argument('--sched', help='LR scheduler type',default='linear')
#parser.add_argument('--gamma', help='LR schedule param',default=1.0, type=float)
parser.add_argument('--nesterov', help='SGD param',default=False, action='store_true')
parser.add_argument('--momentum', help='SGD param',default=0.0, type=float)
#parser.add_argument('--dampening', help='SGD param',default=0.0, type=float)
parser.add_argument('--opt', help='optimizer type',default='adamw')
parser.add_argument('--weight_decay', help='L2 penalty',default=0.0, type=float)
#parser.add_argument('--centercrop', help='crop to square',default=False, action='store_true')
parser.add_argument('--step', help='LR scheduler batches per step',default=1000000000, type=int)
parser.add_argument('--start_factor', help='LR linear schedule parameter',default=1., type=float)
parser.add_argument('--end_factor', help='LR linear schedule parameter',default=1./50, type=float)
parser.add_argument('--total_iters', help='LR linear schedule parameter',default=10, type=float)
parser.add_argument('--seed', help='random seed',default=None, type=int)
args = parser.parse_args()
if args.seed is None:
    random.seed(None) # random seed
    args.seed = random.randint(0,1000000000)
random.seed(args.seed)
args.rng = np.random.default_rng(args.seed)
#np.random.RandomState.seed(np.random.MT19937(np.random.SeedSequence(args.seed)))
#args.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(args.seed)))
#print(args)

if args.log is None:
    if not os.path.exists('log'):
        os.makedirs('log')
    args.date = subprocess.check_output(['/usr/bin/date', '+%Y.%m.%d-%H.%M.%S'])
    args.date = args.date.decode("utf-8")
    args.date = args.date.rstrip()
    args.log = 'log/log.{}'.format(args.date)
print(args)
with open(args.log, 'a') as f:
    print('ARGS',args,file=f)

# --------------------------------------------------------------------------------------------------------------------------
if args.mode=='finetune_resnet18' or args.mode=='pretrain':
    resnet18 = timm.create_model('resnet18.a1_in1k', pretrained=True, features_only=True)
    for param in resnet18.parameters():
        param.requires_grad = False
    resnet18 = resnet18.eval()
    data_config = timm.data.resolve_model_data_config(resnet18)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    print('resnet18 image encoder model loaded')
    if args.verbose:
        s=torchinfo.summary(resnet18,col_names=["input_size","output_size","num_params"],input_size=(1,3,224,224))
        print('RESNET-18',s)
        with open(args.log, 'a') as f:
            print('RESNET-18',s,file=f)
    resnet18 = resnet18.to(args.device)

# --------------------------------------------------------------------------------------------------------------------------
if args.mode=='finetune_encoder' or args.mode=='pretrain':
    class IE120R(
        nn.Module,
        PyTorchModelHubMixin, 
        repo_url="https://github.com/siliconperception/models",
        license="mit",
    ):
        def __init__(self,alt="large"):
            super().__init__()
            if alt=='large':
                self.layer1  = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU())
                self.layer2  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU())
                self.layer3  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU())
                self.layer4  = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.layer5  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.layer6  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.layer7  = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.layer8  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.layer9  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.layerf0  = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
                self.layer10 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.layer11 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.layer12 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.layerf1  = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
                self.layer13 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
                self.layer14 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
                self.layer15 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
                self.layerf2  = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
                self.layer16 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU())
                self.layer17 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())
                self.layer18 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())
                self.layerf3  = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
                self.layer19 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(1024), nn.ReLU())
                self.layer20 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(1024), nn.ReLU())
                self.layer21 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(1024), nn.ReLU())
                self.layerf4  = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
            if alt=='medium':
                self.layer1  = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU())
                self.layer2  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU())
                self.layer3  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU())
                self.layer4  = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.layer5  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.layer6  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.layer7  = nn.Sequential(nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU())
                self.layer8  = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(96), nn.ReLU())
                self.layer9  = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(96), nn.ReLU())
                self.layerf0  = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
                self.layer10 = nn.Sequential(nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.layer11 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.layer12 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.layerf1  = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
                self.layer13 = nn.Sequential(nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(192), nn.ReLU())
                self.layer14 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU())
                self.layer15 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU())
                self.layerf2  = nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0)
                self.layer16 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(384), nn.ReLU())
                self.layer17 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU())
                self.layer18 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU())
                self.layerf3  = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
                self.layer19 = nn.Sequential(nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(768), nn.ReLU())
                self.layer20 = nn.Sequential(nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(768), nn.ReLU())
                self.layer21 = nn.Sequential(nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(768), nn.ReLU())
                self.layerf4  = nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0)
            if alt=='small':
                self.layer1  = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU())
                self.layer2  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU())
                self.layer3  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU())
                self.layer4  = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.layer5  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.layer6  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.layer7  = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.layer8  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.layer9  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.layerf0  = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
                self.layer10 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.layer11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.layer12 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.layerf1  = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
                self.layer13 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.layer14 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.layer15 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.layerf2  = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
                self.layer16 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
                self.layer17 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
                self.layer18 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
                self.layerf3  = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
                self.layer19 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU())
                self.layer20 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())
                self.layer21 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())
                self.layerf4  = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)
            out = self.layer7(out)
            out = self.layer8(out)
            out = self.layer9(out)
            f0 = self.layerf0(out)
            out = self.layer10(out)
            out = self.layer11(out)
            out = self.layer12(out)
            f1 = self.layerf1(out)
            out = self.layer13(out)
            out = self.layer14(out)
            out = self.layer15(out)
            f2 = self.layerf2(out)
            out = self.layer16(out)
            out = self.layer17(out)
            out = self.layer18(out)
            f3 = self.layerf3(out)
            out = self.layer19(out)
            out = self.layer20(out)
            out = self.layer21(out)
            f4 = self.layerf4(out)
            return f0,f1,f2,f3,f4

    encoder = IE120R(alt=args.alt)
    if args.verbose:
        s=torchinfo.summary(encoder,col_names=["input_size","output_size","num_params"],input_size=(1,3,896,896))
        print('IE120R',s)
        with open(args.log, 'a') as f:
            print('IE120',s,file=f)
    if args.encoder is not None:
        encoder.load_state_dict(torch.load('{}'.format(args.encoder)))
        print('image encoder model state_dict loaded',args.encoder)
    if args.mode=='finetune_encoder':
        for param in encoder.parameters():
            param.requires_grad = False
        encoder = encoder.eval()
    encoder = encoder.to(args.device)
    
# --------------------------------------------------------------------------------------------------------------------------
synlabel={}
labeltext={}
meta = scipy.io.loadmat('{}/imagenet/devkit-1.0/data/meta.mat'.format(args.imagenet))
synsets = meta['synsets']
for i,s in enumerate(synsets):
    synlabel[s[0][1][0]] = s[0][0][0][0]
    labeltext[s[0][0][0][0]] = s[0][2][0]
print('imagenet metadata loaded')
with open('{}/imagenet/train/flist'.format(args.imagenet), 'r') as f:
    train_flist = f.readlines()
print('imagenet training flist loaded',len(train_flist))
train_label = [synlabel[fn[0:9]] for fn in train_flist]
print('imagenet training labels loaded',len(train_label))
with open('{}/imagenet/val/flist'.format(args.imagenet), 'r') as f:
    val_flist = f.readlines()
print('imagenet flist loaded - VALIDATION DISTRIBUTION',len(val_flist))
with open('{}/imagenet/devkit-1.0/data/ILSVRC2010_validation_ground_truth.txt'.format(args.imagenet), 'r') as f:
    val_label = [int(line) for line in f.readlines()]
print('imagenet labels loaded - VALIDATION DISTRIBUTION',len(val_label))

def sample_img(dist='train'):
    # return center-cropped image from {train,val} distribution
    if dist=='train':
        flist=train_flist
        label=train_label
    if dist=='val':
        flist=val_flist
        label=val_label
    while True:
        i = np.random.randint(len(flist))
        img = cv2.imread('{}/imagenet/{}/{}'.format(args.imagenet,dist,flist[i].rstrip()))
        if img is not None:
            # center crop
            side = min(img.shape[0],img.shape[1])
            img = img[img.shape[0]//2-side//2:img.shape[0]//2+side//2,img.shape[1]//2-side//2:img.shape[1]//2+side//2]
            break
    return img,label[i]

def transform_img(img,dmean=0.0,dstd=1.0):
    d = np.divide(img,255.)
    d = np.subtract(d,dmean)
    d = np.divide(d,dstd)
    d = np.rollaxis(d,-1,0)
    return d

#for k in range(20):
#    img,label = sample_img()
#    #i = np.random.randint(len(train_flist))
#    #print('i',i,'flist',train_flist[i],'label',train_label[i],'text',labeltext[train_label[i]])
#    #img = cv2.imread('{}/imagenet/{}/{}'.format(args.imagenet,'train',train_flist[i].rstrip()))
#    cv2.imshow('resnet',img)
#    print('label',label,'text',labeltext[label],'img.shape',img.shape)
#    if cv2.waitKey(0)==120: # 'x'
#        exit()

# --------------------------------------------------------------------------------------------------------------------------
def worker(stop,q,args,dist,dmean,dstd,roi,labeltype='imagenet'):
    while not stop.is_set():
        d = np.zeros([args.batch,3,roi,roi],dtype=np.float32)
        dimg=[]
        if labeltype=='imagenet':
            l = np.zeros([args.batch],dtype=np.int64)
        for i in range(args.batch):
            img,label = sample_img(dist)
            dimg.append(img)
            d[i]=transform_img(cv2.resize(img,dsize=(roi,roi),interpolation=cv2.INTER_CUBIC),dmean,dstd)
            if labeltype=='imagenet':
                l[i]=label-1
        if labeltype=='resnet18':
            dref = [transform_img(cv2.resize(img,dsize=(224,224),interpolation=cv2.INTER_CUBIC),dmean,dstd) for img in dimg]
            dref = np.array(dref,dtype=np.float32)
            dref=torch.utils.data.default_convert(dref)
            dref = dref.to(args.device)
            l = resnet18(dref)
        q.put((d,l))

stop = threading.Event()
stop.clear()
workers=[]
if args.mode=='finetune_resnet18':
    dmean = transforms.transforms[-1].mean.numpy()
    dstd = transforms.transforms[-1].std.numpy()
    roi=224
    labeltype='imagenet'
if args.mode=='finetune_encoder':
    dmean = 0
    dstd = 1
    roi=896
    labeltype='imagenet'
if args.mode=='pretrain':
    dmean = transforms.transforms[-1].mean.numpy()
    dstd = transforms.transforms[-1].std.numpy()
    roi=896
    labeltype='resnet18'
q0 = queue.Queue(maxsize=args.workers)
for _ in range(args.workers):
    w = threading.Thread(target=worker, args=[stop,q0,args,'train',dmean,dstd,roi,labeltype], daemon=False)
    w.start()
    workers.append(w)
q1 = queue.Queue(maxsize=args.workers)
for _ in range(args.workers):
    w = threading.Thread(target=worker, args=[stop,q1,args,'val',dmean,dstd,roi,labeltype], daemon=False)
    w.start()
    workers.append(w)

# --------------------------------------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, encoder):
        super(Decoder, self).__init__()
        self.encoder = encoder
        self.layerp = nn.Conv2d(512, 1000, kernel_size=7, stride=1) # linear projection from final 7x7 feature map to 1000 imagenet classes

    def forward(self, x):
        fmap = self.encoder(x)[4]
        y = self.layerp(fmap)
        return y[:,:,0,0]

if args.mode=='finetune_resnet18':
    model = Decoder(resnet18)
if args.mode=='finetune_encoder':
    model = Decoder(encoder)
if args.mode=='pretrain':
    model = encoder
device = torch.device(args.device)
model = model.to(device)

if args.opt=='adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
if args.opt=='sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov,dampening=0.0)
classify = nn.CrossEntropyLoss() # we will use class indices for softmax loss
abs_loss = torch.nn.L1Loss(reduce='mean')
mse_loss = torch.nn.MSELoss(reduction='mean')
cos_loss = torch.nn.CosineSimilarity(dim=-3)
pdist = nn.PairwiseDistance(p=2,keepdim=True)

if args.sched=='linear':
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.start_factor, end_factor=args.end_factor, total_iters=args.total_iters)
if args.sched=='cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.tmax, args.eta_min)
i=0
larr=[] # loss
garr=[] # gradient
aarr=[0] # accuracy
while i<1+args.nbatch:
    cv2.waitKey(1)
    i+=1
    (x,y)=q0.get()
    if args.show and (i%100)==0:
        img,_ = sample_img('train')
        cv2.imshow('show', cv2.resize(img,dsize=(roi,roi),interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(100)
    #print('x0',np.amin(x0),np.amax(x0),np.mean(x0))
    #print('y0',np.amin(y0),np.amax(y0),np.mean(y0))
    x=torch.utils.data.default_convert(x)
    x = x.to(device)
    model.train()
    o = model(x)
    if args.mode=='finetune_resnet18' or args.mode=='finetune_encoder':
        y=torch.utils.data.default_convert(y)
        y = y.to(device)
        loss = classify(o,y)
    if args.mode=='pretrain':
        if args.loss=='tbd':
            loss=0
            for j in range(5):
                #loss += mse_loss(o[j],y[j])
                #loss += torch.mean(torch.acos(cos_loss(o[j],y[j]))/np.pi)
                #loss += torch.mean(pdist(torch.reshape(o[j],[args.batch,-1]), torch.reshape(y[j],[args.batch,-1])))
                loss += torch.mean(pdist(torch.flatten(o[j]),torch.flatten(y[j])))
            #loss += mse_loss(o[0],y[0])
            #loss += mse_loss(o[1],y[1])
            #loss += mse_loss(o[2],y[2])
            #loss += mse_loss(o[3],y[3])
            #loss += mse_loss(o[4],y[4])
            #loss += torch.mean(torch.acos(cos_loss(o[0],y[0]))/np.pi)
            #loss += torch.mean(torch.acos(cos_loss(o[1],y[1]))/np.pi)
            #loss += torch.mean(torch.acos(cos_loss(o[2],y[2]))/np.pi)
            #loss += torch.mean(torch.acos(cos_loss(o[3],y[3]))/np.pi)
            #loss += torch.mean(torch.acos(cos_loss(o[4],y[4]))/np.pi)
        if args.loss=='mse':
            loss = mse_loss(o[0],y[0])+mse_loss(o[1],y[1])+mse_loss(o[2],y[2])+mse_loss(o[3],y[3])+mse_loss(o[4],y[4])
        if args.loss=='cos':
            loss=0
            loss += torch.mean(torch.acos(cos_loss(o[0],y[0]))/np.pi)
            loss += torch.mean(torch.acos(cos_loss(o[1],y[1]))/np.pi)
            loss += torch.mean(torch.acos(cos_loss(o[2],y[2]))/np.pi)
            loss += torch.mean(torch.acos(cos_loss(o[3],y[3]))/np.pi)
            loss += torch.mean(torch.acos(cos_loss(o[4],y[4]))/np.pi)
            loss += abs_loss(torch.std(o[0],dim=-3),torch.std(y[0],dim=-3))
            loss += abs_loss(torch.std(o[1],dim=-3),torch.std(y[1],dim=-3))
            loss += abs_loss(torch.std(o[2],dim=-3),torch.std(y[2],dim=-3))
            loss += abs_loss(torch.std(o[3],dim=-3),torch.std(y[3],dim=-3))
            loss += abs_loss(torch.std(o[4],dim=-3),torch.std(y[4],dim=-3))
            #loss += abs_loss(torch.mean(o[0],dim=-3),torch.mean(y[0],dim=-3))
            #loss += abs_loss(torch.mean(o[1],dim=-3),torch.mean(y[1],dim=-3))
            #loss += abs_loss(torch.mean(o[2],dim=-3),torch.mean(y[2],dim=-3))
            #loss += abs_loss(torch.mean(o[3],dim=-3),torch.mean(y[3],dim=-3))
            #loss += abs_loss(torch.mean(o[4],dim=-3),torch.mean(y[4],dim=-3))


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    larr.append(loss.item())
    lavg = np.mean(larr[-args.avg:])
    if (i%args.step)==0:
        scheduler.step()

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
    aavg = np.mean(aarr[-args.avg:])
    if args.mode=='finetune_resnet18' or args.mode=='finetune_encoder':
        s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} opt {:15} accuracy {:12.6f}'.format(
            i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,args.opt+'_'+args.loss+'_'+args.sched,aavg)
    if args.mode=='pretrain':
        s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} opt {:15} mse {:12.6f} f4 {:12.6f} {:12.6f} y4 {:12.6f} {:12.6f}'.format(
            i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,args.opt+'_'+args.loss+'_'+args.sched,torch.nn.functional.mse_loss(o[4],y[4]),torch.mean(o[4]),torch.std(o[4]),torch.mean(y[4]),torch.std(y[4]))

    print(s)
    with open(args.log, 'a') as f:
        print(s,file=f)
    
    if args.save is not None and ((i-args.saveoffset)%args.saveinterval)==0:
        torch.save(encoder.state_dict(), '{}'.format(args.save))
        if args.push:
            encoder.save_pretrained('ie120r_medium_{}'.format(siliconperception.__version__))
            encoder.push_to_hub("siliconperception/IE120R")

if args.mode=='finetune_resnet18' or args.mode=='finetune_encoder':
    model.eval()
    for j in range(50000//args.batch):
        (x0,y0)=q1.get()
        x=torch.utils.data.default_convert(x0)
        x = x.to(device)
        logits = model(x)
        prob = torch.nn.functional.softmax(logits,dim=-1)
        prob = prob.cpu().detach().numpy()
        pred = np.argmax(prob,axis=-1)
        acc = np.mean(pred==y0)
#            if (j%10)==0:
#                print('TEST',j,acc)
#                #print('prob',prob,'pred',pred,'l',l)
        aarr.append(acc)
    print('ACCURACY',np.mean(aarr))

print('STOPPING WORKERS')
stop.set()
for w in workers:
    while not q0.empty():
        q0.get()
    while not q1.empty():
        q1.get()
    w.join()
print('EXIT MAIN')
exit()




















exit()
#from transformers import AutoImageProcessor, AutoModelForImageClassification
#from datasets import load_dataset

#dataset = load_dataset("huggingface/cats-image")
#image = dataset["test"]["image"][0]
#x = x.to('cpu')

#resnet18 = timm.create_model( 'resnet18.a1_in1k', pretrained=True, features_only=True,)
#resnet18 = resnet18.eval()
##data_config = timm.data.resolve_model_data_config(model)
##transforms = timm.data.create_transform(**data_config, is_training=False)
##output = model(torch.utils.data.default_convert(transforms(img).unsqueeze(0)))  # unsqueeze single image into batch of 1
#x = np.random.uniform(0,1,size=(1,3,224,224)).astype(np.float32)
#output = resnet18(torch.utils.data.default_convert(x))
#torchinfo.summary(resnet18,col_names=["input_size","output_size","num_params"],input_size=(1,3,224,224),device='cpu')
#for o in output:
#    # print shape of each feature map in output
#    # e.g.:
#    #  torch.Size([1, 64, 112, 112])
#    #  torch.Size([1, 64, 56, 56])
#    #  torch.Size([1, 128, 28, 28])
#    #  torch.Size([1, 256, 14, 14])
#    #  torch.Size([1, 512, 7, 7])
#    print(o.shape)

class IE120R(
    nn.Module,
    PyTorchModelHubMixin, 
    #repo_url="https://github.com/siliconperception/models",
    license="mit",
):
    def __init__(self):
        super().__init__()
        #self.layers=[]
        #self.layers.extend(nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2), nn.BatchNorm2d(16), nn.ReLU()))
        #self.layers.extend(nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1), nn.BatchNorm2d(16), nn.ReLU()))
        #self.layers.extend(nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1), nn.BatchNorm2d(16), nn.ReLU()))
        #self.layers.extend(nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1), nn.BatchNorm2d(16), nn.ReLU()))
        #self.layer1  = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU())

        self.layer1  = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.layer2  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.layer3  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.layer4  = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer5  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer6  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer7  = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer8  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer9  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer10 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer12 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer13 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer14 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer15 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer16 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer17 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer18 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer19 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.layer20 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.layer21 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())

        self.layerf0  = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.layerf1  = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.layerf2  = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.layerf3  = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.layerf4  = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        f0 = self.layerf0(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        f1 = self.layerf1(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        f2 = self.layerf2(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        f3 = self.layerf3(out)
        out = self.layer19(out)
        out = self.layer20(out)
        out = self.layer21(out)
        f4 = self.layerf4(out)
        return f0,f1,f2,f3,f4

#        #out=x
#        #for l in self.layers:
#        #    out = l(out)
#        #return out
#        out = torch.nn.functional.relu(self.layer1(x))
#        out = torch.nn.functional.relu(self.layer2(out))
#        out = torch.nn.functional.relu(self.layer3(out))
#        out = torch.nn.functional.relu(self.layer4(out))
#        out = torch.nn.functional.relu(self.layer5(out))
#        out = torch.nn.functional.relu(self.layer6(out))
#        out = torch.nn.functional.relu(self.layer7(out))
#        out = torch.nn.functional.relu(self.layer8(out))
#        f0 = self.layer9(out)
#        out = torch.nn.functional.relu(f0)
#        out = torch.nn.functional.relu(self.layer10(out))
#        out = torch.nn.functional.relu(self.layer11(out))
#        f1 = self.layer12(out)
#        out = torch.nn.functional.relu(f1)
#        out = torch.nn.functional.relu(self.layer13(out))
#        out = torch.nn.functional.relu(self.layer14(out))
#        f2 = self.layer15(out)
#        out = torch.nn.functional.relu(f2)
#        out = torch.nn.functional.relu(self.layer16(out))
#        out = torch.nn.functional.relu(self.layer17(out))
#        f3 = self.layer18(out)
#        out = torch.nn.functional.relu(f3)
#        out = torch.nn.functional.relu(self.layer19(out))
#        out = torch.nn.functional.relu(self.layer20(out))
#        f4 = self.layer21(out)
#        return f0,f1,f2,f3,f4

    def fuse_conv_and_bn(self,conv,bn):
        fusedconv = torch.nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=True)
        w_eff = conv.weight.clone()
        b_eff = conv.bias.clone()
        with torch.no_grad():
            for i in range(conv.out_channels):
                w_eff[i,:,:,:] = torch.div(torch.mul(bn.weight[i],w_eff[i,:,:,:]),torch.sqrt(bn.running_var[i]+bn.eps))
                b_eff[i] = torch.add(torch.div(torch.mul(bn.weight[i],torch.sub(b_eff[i],bn.running_mean[i])),torch.sqrt(bn.running_var[i]+bn.eps)),bn.bias[i])
        fusedconv.weight.data = w_eff.data
        fusedconv.bias.data = b_eff.data
        return fusedconv

    def fuse_save(self, fn):
        # merge BN with previous Conv2D
        enc = nn.Sequential(OrderedDict([
          ('conv1', self.fuse_conv_and_bn(self.layer1[0],self.layer1[1])), ('relu1', nn.ReLU()),
          ('conv2', self.fuse_conv_and_bn(self.layer2[0],self.layer2[1])), ('relu2', nn.ReLU()),
          ('conv3', self.fuse_conv_and_bn(self.layer3[0],self.layer3[1])), ('relu3', nn.ReLU()),
          ('conv4', self.fuse_conv_and_bn(self.layer4[0],self.layer4[1])), ('relu4', nn.ReLU()),
          ('conv5', self.fuse_conv_and_bn(self.layer5[0],self.layer5[1])), ('relu5', nn.ReLU()),
          ('conv6', self.fuse_conv_and_bn(self.layer6[0],self.layer6[1])), ('relu6', nn.ReLU()),
          ('conv7', self.fuse_conv_and_bn(self.layer7[0],self.layer7[1])), ('relu7', nn.ReLU()),
          ('conv8', self.fuse_conv_and_bn(self.layer8[0],self.layer8[1])), ('relu8', nn.ReLU()),
          ('conv9', self.fuse_conv_and_bn(self.layer9[0],self.layer9[1])), ('relu9', nn.ReLU()),
          ('conv10', self.fuse_conv_and_bn(self.layer10[0],self.layer10[1])), ('relu10', nn.ReLU()),
          ('conv11', self.fuse_conv_and_bn(self.layer11[0],self.layer11[1])), ('relu11', nn.ReLU()),
          ('conv12', self.fuse_conv_and_bn(self.layer12[0],self.layer12[1])), ('relu12', nn.ReLU()),
          ('conv13', self.fuse_conv_and_bn(self.layer13[0],self.layer13[1])), ('relu13', nn.ReLU()),
          ('conv14', self.fuse_conv_and_bn(self.layer14[0],self.layer14[1])), ('relu14', nn.ReLU()),
          ('conv15', self.fuse_conv_and_bn(self.layer15[0],self.layer15[1])), ('relu15', nn.ReLU()),
          ('conv16', self.fuse_conv_and_bn(self.layer16[0],self.layer16[1])), ('relu16', nn.ReLU()),
          ('conv17', self.fuse_conv_and_bn(self.layer17[0],self.layer17[1])), ('relu17', nn.ReLU()),
          ('conv18', self.layer18[0]),
        ]))
        # save fused PyTorch model to file for processing by CNN to Verilog compiler (perceptron.py)
        torch.save(enc, fn)

encoder = IE120R()
#encoder.cuda()
#print('cuda',torch.cuda.is_available())
#print('encoder',dir(encoder),encoder)
#torchinfo.summary(encoder,col_names=["input_size","output_size","num_params"],input_size=(1,3,896,896),device='cpu')
#encoder.fuse_save('stub.pt')


#image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
#model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
#print('model',dir(model))
#print('resnet',dir(model.resnet))
#inputs = image_processor(image, return_tensors="pt")
#with torch.no_grad():
#    o = model(**inputs)
#    print('o',o,dir(o))
#    print('logits',o.logits.shape)
#    print('hidden_states',o.hidden_states.shape)
#    exit()
#    logits = model(**inputs).logits
## model predicts one of the 1000 ImageNet classes
#predicted_label = logits.argmax(-1).item()
#print(model.config.id2label[predicted_label])
#exit()


#encoder = IE120NX()
#if args.stub:
#    torchinfo.summary(encoder,col_names=["input_size","output_size","num_params"],input_size=(1,3,768,768))
#    encoder.fuse_save('stub.pt')
#    exit()
#s=torchinfo.summary(encoder,col_names=["input_size","output_size","num_params"],input_size=(1,3,896,896),device='cpu')
#with open(args.log, 'a') as f:
#    print('ENCODER',s,file=f)

encoder = encoder.to(args.device)
if args.encoder is not None:
    encoder.load_state_dict(torch.load('{}'.format(args.encoder)))
    print('image encoder model state_dict loaded',args.encoder)

resnet18 = timm.create_model( 'resnet18.a1_in1k', pretrained=True, features_only=True,)
#ref = timm.create_model( 'convnextv2_tiny.fcmae', pretrained=True, features_only=True,)
for param in resnet18.parameters():
    param.requires_grad = False
resnet18 = resnet18.eval()
data_config = timm.data.resolve_model_data_config(resnet18)
transforms = timm.data.create_transform(**data_config, is_training=False)
print('resnet18 image encoder model loaded')
#s=torchinfo.summary(resnet18,col_names=["input_size","output_size","num_params"],input_size=(1,3,224,224))
#with open(args.log, 'a') as f:
#    print('RESNET-18',s,file=f)

class Batch:
    def __init__(self,args,ref,transforms=None,dist='train',head='ref'):
        self.dist=dist
        self.head=head
        self.origin=0
        self.ref=ref
        self.transforms=transforms
        self.synlabel={}
        self.labeltext={}
        meta = scipy.io.loadmat('{}/imagenet/devkit-1.0/data/meta.mat'.format(args.imagenet))
        synsets = meta['synsets']
        for i,s in enumerate(synsets):
            self.synlabel[s[0][1][0]] = s[0][0][0][0]
            self.labeltext[s[0][0][0][0]] = s[0][2][0]
        print('imagenet metadata loaded')
        
        if dist=='train':
            with open('{}/imagenet/train/flist'.format(args.imagenet), 'r') as f:
                self.flist = f.readlines()
            print('imagenet training flist loaded',len(self.flist))
            random.shuffle(self.flist)
            print('imagenet training flist shuffled')
        if dist=='val':
            with open('{}/imagenet/val/flist'.format(args.imagenet), 'r') as f:
                self.flist = f.readlines()
            print('imagenet flist loaded - VALIDATION DISTRIBUTION',len(self.flist))
            with open('{}/imagenet/devkit-1.0/data/ILSVRC2010_validation_ground_truth.txt'.format(args.imagenet), 'r') as f:
                self.val_label = [int(line) for line in f.readlines()]
            print('imagenet labels loaded - VALIDATION DISTRIBUTION',len(self.val_label))
        #if args.nsamples>0:
        #    self.flist = self.flist[0:args.nsamples]
        #    print('imagenet flist truncated',len(self.flist))
        #self.flist=self.flist[0:2]
        #self.nf = args.nf_init

    #def bump_flist(self):
    #    if self.nf < len(self.flist):
    #        self.nf +=1

    def shuffle(self):
        random.shuffle(self.flist)

    #def slide(self):
    #    self.origin = (self.origin+1) % (len(self.flist)-args.nsamples)

    def generate_batch(self,args,show=False,noise=False):
        dimg = np.zeros([args.batch,args.roi,args.roi,3]).astype(np.uint8)
        if self.head=='ref':
            l = np.zeros([args.batch,512,24,24]).astype(float) # feature map
        sup_label=[]
        for i in range(args.batch):
            while True:
                fn = random.choice(self.flist)
                img = cv2.imread('{}/imagenet/{}/{}'.format(args.imagenet,self.dist,fn.rstrip()))
                if img is not None:
                    if self.dist=='val':
                        sup_label.append(self.val_label[int(fn[15:15+8])-1]-1)
                    if self.dist=='train':
                        sup_label.append(self.synlabel[fn[0:9]]-1)
                    break
            # center crop
            side1 = min(img.shape[0],img.shape[1])
            img = img[img.shape[0]//2-side1//2:img.shape[0]//2+side1//2,img.shape[1]//2-side1//2:img.shape[1]//2+side1//2]
            img = cv2.resize(img,dsize=(args.roi,args.roi),interpolation=cv2.INTER_CUBIC)
            dimg[i]=img

        if noise:
            d = args.rng.integers(0,256,d.shape,d.dtype)
        if show:
            cv2.imshow('ref', dimg[args.rng.integers(args.batch)])
        if self.head=='ref':
            dref = [cv2.resize(img,dsize=(224,224),interpolation=cv2.INTER_CUBIC) for img in dimg]
            dref = np.array(dref)
            dref = dref/255.
            if self.transforms is not None:
                dmean = self.transforms.transforms[-1].mean.numpy()
                dstd = self.transforms.transforms[-1].std.numpy()
                #print('dmean',dmean,'dstd',dstd)
                dref = np.subtract(dref,dmean)
                dref = np.divide(dref,dstd)
            dref = np.rollaxis(dref,-1,1)
            dref = dref.astype(np.float32)
            dref=torch.utils.data.default_convert(dref)
            dref = dref.to(args.device)
            l = self.ref(dref)
        if self.head=='sup':
            l = np.array(sup_label,dtype=int)

        d = dimg/255.
        if self.transforms is not None:
            dmean = self.transforms.transforms[-1].mean.numpy()
            dstd = self.transforms.transforms[-1].std.numpy()
            d = np.subtract(d,dmean)
            d = np.divide(d,dstd)
        d = np.rollaxis(d,-1,1)
        d = d.astype(np.float32)

        return d,l
    
if args.pretrain:
    dataset = Batch(args,resnet18,transforms,dist='train',head='ref')
    
    encoder.train() # train mode
    if args.loss=='cos':
        abs_loss = torch.nn.L1Loss(reduce='mean')
        mse_loss = torch.nn.MSELoss(reduction='mean')
        cos_loss = torch.nn.CosineSimilarity(dim=-3)
    if args.loss=='softmax':
        bce_loss = torch.nn.BCELoss(reduction='sum')
    if args.loss=='bce':
        bce_loss = torch.nn.BCELoss(reduction='mean')
    if args.loss=='hyb':
        #bce_loss = torch.nn.BCELoss(reduction='mean')
        mse_loss = torch.nn.MSELoss(reduction='mean')
        abs_loss = torch.nn.L1Loss(reduction='mean')
        #xen_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        cos_loss = torch.nn.CosineSimilarity(dim=-3)
    if args.loss=='mse':
        mse_loss = torch.nn.MSELoss(reduction='mean')
    if args.loss=='crossentropy':
        loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
    if args.loss=='smooth':
        abs_loss = torch.nn.SmoothL1Loss(reduce='mean')
    if args.loss=='abs':
        abs_loss = torch.nn.L1Loss(reduce='mean')
    if args.opt=='adamw':
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    if args.opt=='sgd':
        optimizer = torch.optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov,dampening=0.0)
    
    if args.sched=='warmup':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.start_factor, end_factor=args.end_factor, total_iters=args.total_iters)
    if args.sched=='linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.start_factor, end_factor=args.end_factor, total_iters=args.total_iters)
    if args.sched=='plat':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=args.factor,min_lr=args.min_lr,patience=args.patience)
    if args.sched=='grad':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=args.factor,min_lr=args.min_lr)
    if args.sched=='multi':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.gamma)
    if args.sched=='cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=args.lr*0.001)
    if args.sched=='inv':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1./np.random.uniform(1,args.lr2))
    if args.sched=='gauss':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: np.random.normal(1,args.lr2))
    if args.sched=='set':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: random.choice([1,0.1,0.01,0.001]))

    
    def worker(stop, q, args, dataset):
        while not stop.is_set():
            d,l = dataset.generate_batch(args)
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
    larr1=[]
    larr2=[]
    larr3=[]
    garr=[]
    device = torch.device(args.device)
    while i<1+args.nbatch:
        (x0,y)=q.get()
        #print('y0',y0.shape,'min',np.min(y0),'max',np.max(y0),'mean',np.mean(y0),'std',np.std(y0))
        #print(np.histogram(y0))
        #lm = np.sign(np.mean(y0,axis=-3))
        #lm = np.mean(y0,axis=-3)
        #print('lm',lm.shape,lm)
        #exit()
        x=torch.utils.data.default_convert(x0)
        x = x.to(device)
        #y0=torch.utils.data.default_convert(y[0]).to(device)
        #y1=torch.utils.data.default_convert(y[1]).to(device)
        #y2=torch.utils.data.default_convert(y[2]).to(device)
        #y3=torch.utils.data.default_convert(y[3]).to(device)
        #y4=torch.utils.data.default_convert(y[4]).to(device)
        y0=y[0]
        y1=y[1]
        y2=y[2]
        y3=y[3]
        y4=y[4]
        fmap = encoder(x)
        f0=fmap[0]
        f1=fmap[1]
        f2=fmap[2]
        f3=fmap[3]
        f4=fmap[4]
        #foo = torch.nn.functional.cosine_similarity(fmap,fmap,dim=-3)
        #print('foo',foo.shape)
        #print(foo[0])
        #exit()
        #print('fmap',fmap.shape)
        #ref = mamba(x)[1][3]
        #ref_np = ref.detach().cpu().numpy()
        ##ref_np = np.clip(ref_np,a_min=None,a_max=100)
        ##ref_np = np.exp(ref_np)
        #c,e = np.histogram(ref_np,bins=100,density=False)
        #print('mean',np.mean(ref_np),'std',np.std(ref_np))
        #for k in range(len(c)):
        #    print('c {:6.12f} e {:6.12f}'.format(c[k],e[k]))
        if args.loss=='cos':
            #loss = torch.mean(1-cos_loss(fmap,y))
            #loss = torch.mean(1-cos_loss(f0,y0)) + torch.mean(1-cos_loss(f1,y1)) + torch.mean(1-cos_loss(f2,y2)) + torch.mean(1-cos_loss(f3,y3)) + torch.mean(1-cos_loss(f4,y4)) + abs_loss(f0,y0) + abs_loss(f1,y1) + abs_loss(f2,y2) + abs_loss(f3,y3) + abs_loss(f4,y4)
            loss = torch.mean(torch.acos(cos_loss(f0,y0)))
            loss += torch.mean(torch.acos(cos_loss(f1,y1)))
            loss += torch.mean(torch.acos(cos_loss(f2,y2)))
            loss += torch.mean(torch.acos(cos_loss(f3,y3)))
            loss += torch.mean(torch.acos(cos_loss(f4,y4)))

            loss += abs_loss(torch.std(f0,dim=-3),torch.std(y0,dim=-3))
            loss += abs_loss(torch.std(f1,dim=-3),torch.std(y1,dim=-3))
            loss += abs_loss(torch.std(f2,dim=-3),torch.std(y2,dim=-3))
            loss += abs_loss(torch.std(f3,dim=-3),torch.std(y3,dim=-3))
            loss += abs_loss(torch.std(f4,dim=-3),torch.std(y4,dim=-3))

            #loss = torch.mean(1-cos_loss(f0,y0))
            #loss += torch.mean(1-cos_loss(f1,y1))
            #loss += torch.mean(1-cos_loss(f2,y2))
            #loss += torch.mean(1-cos_loss(f3,y3))
            #loss += torch.mean(1-cos_loss(f4,y4))

            #loss += abs_loss(torch.linalg.vector_norm(f0, ord=1, dim=-3),torch.linalg.vector_norm(y0, ord=2, dim=-3))
            #loss += abs_loss(torch.linalg.vector_norm(f1, ord=1, dim=-3),torch.linalg.vector_norm(y1, ord=2, dim=-3))
            #loss += abs_loss(torch.linalg.vector_norm(f2, ord=1, dim=-3),torch.linalg.vector_norm(y2, ord=2, dim=-3))
            #loss += abs_loss(torch.linalg.vector_norm(f3, ord=1, dim=-3),torch.linalg.vector_norm(y3, ord=2, dim=-3))
            #loss += abs_loss(torch.linalg.vector_norm(f4, ord=1, dim=-3),torch.linalg.vector_norm(y4, ord=2, dim=-3))

            #loss = torch.mean(1-cos_loss(fmap,y)) + torch.mean(torch.abs(torch.std(fmap)-torch.std(y)))
            #loss = torch.mean(1-cos_loss(fmap,y))+mse_loss(fmap,y)
            #loss = torch.mean(1-cos_loss(fmap,y))+abs_loss(fmap,y)
            #loss = torch.mean(torch.abs(torch.sub(cos_loss(fmap,y),1)))+abs_loss(fmap,y)
            #print('loss',loss,type(loss),dir(loss))
            #cos_loss = torch.square(torch.subtract(torch.nn.CosineSimilarity(dim=-3),1.0))
            #loss = cos_loss(fmap,y) + abs_loss(fmap,y)
        if args.loss=='softmax':
            loss = bce_loss(torch.nn.functional.softmax(fmap,dim=-3),torch.nn.functional.softmax(y,dim=-3))
        if args.loss=='abs':
            loss = abs_loss(fmap,y)
        if args.loss=='bce':
            loss = bce_loss(torch.nn.functional.sigmoid(fmap),torch.nn.functional.sigmoid(y))
        if args.loss=='hyb':
            loss = torch.mean(1-cos_loss(fmap,y))+mse_loss(fmap,y)
            #loss = mse_loss(fmap,y)+abs_loss(fmap,y)
            #loss1 = bce_loss(torch.nn.functional.sigmoid(fmap),torch.nn.functional.sigmoid(y))
            #loss1 = xen_loss(fmap,torch.nn.functional.sigmoid(y))
            #loss2 = mse_loss(fmap,y)
            #loss3 = abs_loss(fmap,y)
            #loss = loss1+loss2+loss3
            #loss = bce_loss(torch.nn.functional.sigmoid(fmap),torch.nn.functional.sigmoid(y))+mse_loss(fmap,y)+abs_loss(fmap,y)
            #loss = bce_loss(torch.nn.functional.sigmoid(fmap),torch.nn.functional.sigmoid(y))+abs_loss(fmap,y)
        if args.loss=='mse':
            loss = mse_loss(f0,y0)+mse_loss(f1,y1)+mse_loss(f2,y2)+mse_loss(f3,y3)+mse_loss(f4,y4)
        if args.loss=='crossentropy':
            #std = torch.std(ref)
            #ref = torch.divide(ref,std)
            #print('ref 0 min',torch.min(ref),'max',torch.max(ref),'mean',torch.mean(ref),'std',torch.std(ref))
            #ref = (ref+args.prec*std)/(2*args.prec*std)
            #print('ref 1 min',torch.min(ref),'max',torch.max(ref),'mean',torch.mean(ref),'std',torch.std(ref))
            #ref = torch.clamp(ref,0,1)
            #print('ref 2 min',torch.min(ref),'max',torch.max(ref),'mean',torch.mean(ref),'std',torch.std(ref))
            #ref = torch.nn.functional.sigmoid(ref)
            loss = loss_function(fmap,torch.nn.functional.sigmoid(y))
        #ref = torch.multiply(ref,args.refscale)
        #print('ref',ref.shape)
        #print('ref',len(ref))
        #print('ref[1]',len(ref[1]))
        #print([x.shape for x in ref[1]])
        #exit()
         
        #print('fmap',fmap.shape,'min',torch.min(fmap[0]),'max',torch.max(fmap[0]),'ref',ref.shape,'min',torch.min(ref[0]),'max',torch.max(ref[0]))
        #loss = loss_function(fmap,ref)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        larr.append(loss.item())
        lavg = np.mean(larr[-args.avg:])
        #if args.loss=='hyb':
        #    larr1.append(loss1.item())
        #    larr2.append(loss2.item())
        #    larr3.append(loss3.item())
        #    lavg1 = np.mean(larr1[-args.avg:])
        #    lavg2 = np.mean(larr2[-args.avg:])
        #    lavg3 = np.mean(larr3[-args.avg:])

    
        # compute gradient
        total_norm = 0
        parameters = [p for p in encoder.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        garr.append(total_norm)
        gavg = np.mean(garr[-args.avg:])

        #if (i%1)==0:
        #    if loss.item() < args.nf_llim:
        #        dataset.bump_flist()
        # LR scheduler
        if (i%args.step)==0:
            if args.sched=='plat':
                scheduler.step(lavg)
            else:
                scheduler.step()
    
        #if (args.shuffle>0) and (i%args.shuffle)==0:
        #    dataset.shuffle()
        #if (args.slide>0) and (gavg>args.slide):
        #if (args.slide>0) and (lavg<args.slide):
        #if (args.slide>0) and (i%(int(args.slide)))==0:
        #    dataset.slide()

        # print batch statistics
        lr = optimizer.param_groups[0]['lr']
        if args.sched=='multi':
            lr = scheduler.get_last_lr()[0]
        #mom = optimizer.param_groups[0]['momentum']
        #if args.loss=='crossentropy':
        #    fmap = torch.nn.functional.sigmoid(fmap)
        #if args.loss=='hyb':
        #    s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} opt {:15} fmap {:12.6f} {:12.6f} ref {:12.6f} {:12.6f} ns {:9d}'.format(
        #    i,datetime.datetime.now(),lr,args.weight_decay,args.batch,lavg1,lavg2,lavg3,total_norm,gavg,args.opt+'_'+args.loss+'_'+args.sched,torch.mean(fmap),torch.std(fmap),np.mean(y0),np.std(y0),args.nsamples)
        s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} opt {:15} fmap {:12.6f} {:12.6f} ref {:12.6f} {:12.6f}'.format(
        i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,args.opt+'_'+args.loss+'_'+args.sched,torch.mean(f4),torch.std(f4),np.mean(y4.detach().cpu().numpy()),np.std(y4.detach().cpu().numpy()))
        print(s)
        with open(args.log, 'a') as f:
            print(s,file=f)
        
        if args.save is not None and (i%args.saveinterval)==0:
            torch.save(encoder.state_dict(), '{}'.format(args.save))
            #encoder.save_pretrained('ie120nx_{}'.format(siliconperception.__version__))
            #encoder.push_to_hub("siliconperception/IE120NX")
    
        if args.show:
            if (i%100)==0:
                dataset.generate_batch(args,show=True)
            cv2.waitKey(1)

        # batch counter
        i+=1
    
    print('STOPPING WORKERS')
    stop.set()
    for w in workers:
        while not q.empty():
            q.get()
        w.join()
    print('EXIT MAIN')
    exit()

if args.finetune:
    # load pretrained encoder
    if args.alt=='resnet18':
        enc=resnet18
    if args.alt=='ie120r':
        enc=encoder
    # freeze encoder weights
    for param in enc.parameters():
        param.requires_grad = False
    enc.eval() # eval mode
    print('image encoder model frozen',args.alt)
    # decoder model
    class Decoder(nn.Module):
        def __init__(self, encoder, alt=None):
            super(Decoder, self).__init__()
            self.encoder = encoder
            self.alt = alt
            self.layerp = nn.Conv2d(512, 1000, kernel_size=7, stride=1) # linear projection from final 7x7 feature map to 1000 imagenet classes
    
        def forward(self, x):
            fmap = self.encoder(x)[4]
            y = self.layerp(fmap)
            return y[:,:,0,0]

    model = Decoder(enc)
    device = torch.device(args.device)
    model = model.to(device)

    dataset = Batch(args,resnet18,transforms,dist='train',head='sup')
    valset = Batch(args,resnet18,transforms,dist='val',head='sup')
    def worker(stop, q, args, dataset):
        while not stop.is_set():
            d,l = dataset.generate_batch(args)
            q.put((d,l))
    
    q = queue.Queue(maxsize=args.workers)
    stop = threading.Event()
    stop.clear()
    workers=[]
    for _ in range(args.workers):
        w = threading.Thread(target=worker, args=[stop,q,args,dataset], daemon=False)
        w.start()
        workers.append(w)
    
    criterion = nn.CrossEntropyLoss() # we will use class indices for softmax loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    i=1
    larr=[] # loss
    garr=[] # gradient
    aarr=[0] # accuracy
    while i<1+args.nbatch:
        (x0,y0)=q.get()
        #print('x0',np.amin(x0),np.amax(x0))
        #print('y0',np.amin(y0),np.amax(y0))
        x=torch.utils.data.default_convert(x0)
        x = x.to(device)
        y=torch.utils.data.default_convert(y0)
        y = y.to(device)
        model.train()
        logits = model(x)
        loss = criterion(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        larr.append(loss.item())
        lavg = np.mean(larr[-args.avg:])
        if args.sched is not None and (i%args.step)==0:
            scheduler.step()
    
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
        try:
            mom = optimizer.param_groups[0]['momentum']
        except:
            mom=0
        aavg = np.mean(aarr[-args.avg:])
        #s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} accuracy {:12.6f} opt {:15} mom {:12.10f} nest {}'.format(
        #    i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,aavg,args.opt,mom,args.nesterov)
        s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} opt {:15} accuracy {:12.6f}'.format(
            i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,args.opt+'_'+args.sched,aavg)
        print(s)
        with open(args.log, 'a') as f:
            print(s,file=f)
        
        #if args.save is not None and (i%1000)==0:
        #    torch.save(model, '{}'.format(args.save))
    
        if (i%1000)==0:
            model.eval()
            for j in range(100):
                d,l = valset.generate_batch(args)
                x=torch.utils.data.default_convert(d)
                x = x.to(device)
                y=torch.utils.data.default_convert(l)
                y = y.to(device)
                logits = model(x)
                prob = torch.nn.functional.softmax(logits,dim=-1)
                prob = prob.cpu().detach().numpy()
                pred = np.argmax(prob,axis=-1)
                acc = np.mean(pred==l)
                if (j%10)==0:
                    print('TEST',j,acc)
                    #print('prob',prob,'pred',pred,'l',l)
                aarr.append(acc)

        # batch counter
        i+=1
    
    print('STOPPING WORKERS')
    stop.set()
    for w in workers:
        while not q.empty():
            q.get()
        w.join()
    print('EXIT MAIN')
    exit()
