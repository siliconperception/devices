import siliconperception ; print('siliconperception',siliconperception.__version__)
from siliconperception.IE120NX import IE120NX
import torch
import torchinfo ; print('torchinfo',torchinfo.__version__)
import numpy as np
import timm
import argparse
import cv2
import scipy
import numpy as np
from collections import namedtuple
import random
import datetime
import queue
import threading
import subprocess
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slide', help='slide training data distribution if gradient greater',default=-1, type=float)
#parser.add_argument('--cycle', help='cycle through samples',default=False, action='store_true')
parser.add_argument('--shuffle', help='shuffle flist every args.shuffle batches',default=-1, type=int)
parser.add_argument('--nsamples', help='size of flist in samples',default=-1, type=int)
parser.add_argument('--lr2', help='learning rate distribution parameter',default=0.1, type=float)
parser.add_argument('--warmup', help='number of batches at initial LR',default=0, type=int)
parser.add_argument('--stub', help='generate stub for perceptron',default=False, action='store_true')
#parser.add_argument('--nf_llim', help='loss threshold for flist bump',default=0.01, type=float)
parser.add_argument('--min_lr', help='minimum learning rate',default=0.0001, type=float)
parser.add_argument('--patience', help='LR param',default=10, type=int)
parser.add_argument('--loss', help='loss function',default='bce')
#parser.add_argument('--prec', help='scaling factor for reference model output in sigma',default=2, type=float)
parser.add_argument('--show', help='display batches',default=False, action='store_true')
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--nbatch', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--lr', help='initial learning rate',default=0.0005, type=float)
parser.add_argument('--device', help='pytorch execution device',default='cuda')
parser.add_argument('--train', help='pretrain image encoder model',default=False, action='store_true')
parser.add_argument('--encoder', help='input encoder model name',default=None)
#parser.add_argument('--alt', help='encoder model alt type',default='vit')
parser.add_argument('--imagenet', help='imagenet dataset base directory',default='../')
parser.add_argument('--batch', help='batch size',default=1, type=int)
parser.add_argument('--roi', help='input image x/y size',default=768, type=int)
parser.add_argument('--x1', help='include 1x1 examples',default=False, action='store_true')
parser.add_argument('--x2', help='include 2x2 examples',default=False, action='store_true')
parser.add_argument('--x3', help='include 3x3 examples',default=False, action='store_true')
parser.add_argument('--resize', help='resize scale',default=0.0, type=float)
parser.add_argument('--workers', help='number of threads for batch generation',default=20, type=int)
parser.add_argument('--save', help='output encoder model name',default=None)
parser.add_argument('--avg', help='moving average window for lr and grad',default=100, type=int)
parser.add_argument('--factor', help='LR schedule param',default=0.1, type=float)
parser.add_argument('--steps', help='LR scheduler steps',action='store', type=int,nargs='*')
parser.add_argument('--sched', help='LR scheduler type',default='linear')
parser.add_argument('--gamma', help='LR schedule param',default=1.0, type=float)
parser.add_argument('--nesterov', help='SGD param',default=False, action='store_true')
parser.add_argument('--momentum', help='SGD param',default=0.0, type=float)
parser.add_argument('--dampening', help='SGD param',default=0.0, type=float)
parser.add_argument('--opt', help='optimizer type',default='adamw')
parser.add_argument('--weight_decay', help='L2 penalty',default=0.0, type=float)
parser.add_argument('--centercrop', help='crop to square',default=False, action='store_true')
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
print(args)

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

encoder = IE120NX()
if args.stub:
    torchinfo.summary(encoder,col_names=["input_size","output_size","num_params"],input_size=(1,3,768,768))
    encoder.fuse_save('stub.pt')
    exit()
s=torchinfo.summary(encoder,col_names=["input_size","output_size","num_params"],input_size=(1,3,768,768))
with open(args.log, 'a') as f:
    print('ENCODER',s,file=f)

if args.encoder is not None:
    encoder.load_state_dict(torch.load('{}'.format(args.encoder)))
    print('image encoder model state_dict loaded',args.encoder)

ref = timm.create_model( 'convnextv2_tiny.fcmae', pretrained=True, features_only=True,)
for param in ref.parameters():
    param.requires_grad = False
ref = ref.eval()
data_config = timm.data.resolve_model_data_config(ref)
transforms = timm.data.create_transform(**data_config, is_training=False)
s=torchinfo.summary(ref,col_names=["input_size","output_size","num_params"],input_size=(1,3,224,224))
with open(args.log, 'a') as f:
    print('CONVNEXT',s,file=f)

class Batch:
    def __init__(self,args,ref,transforms):
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
        
        with open('{}/imagenet/train/flist'.format(args.imagenet), 'r') as f:
            self.flist = f.readlines()
        print('imagenet flist loaded',len(self.flist))
        random.shuffle(self.flist)
        print('imagenet flist shuffled')
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

    def slide(self):
        self.origin = (self.origin+1) % (len(self.flist)-args.nsamples)

    def generate_batch(self,args,show=False,noise=False):
        dimg = np.zeros([args.batch,args.roi,args.roi,3]).astype(np.uint8)
        l = np.zeros([args.batch,640,24,24]).astype(float) # feature map
    
        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
        #choices = []
        #if args.x1:
        #    choices.extend(1*['1x1'])
        #if args.x2:
        #    choices.extend(1*['2x2'])
        #if args.x3:
        #    choices.extend(1*['3x3'])
    
        #choices = ['1x1','2x2','3x3']
        choices = ['1x1','1x1','1x1']
        for i in range(args.batch):
            #sc = random.choice(choices)
            sc = choices[i%3] # equal sample 3 different scales
            ir=[] # image rectangles
            roi=args.roi
            if sc=='1x1':
                ir.append(Rectangle(0, 0, roi, roi))
            if sc=='2x2':
                s = (roi*args.resize)//2
                x0 = args.rng.integers((roi//2)-s,(roi//2)+s+1)
                y0 = args.rng.integers((roi//2)-s,(roi//2)+s+1)
                ir.append(Rectangle(0, 0, x0, y0))
                ir.append(Rectangle(x0, 0, roi, y0))
                ir.append(Rectangle(0, y0, x0, roi))
                ir.append(Rectangle(x0, y0, roi, roi))
            if sc=='3x3':
                s = (roi*args.resize)//3
                x0 = args.rng.integers(1*(roi//3)-s,1*(roi//3)+s+1)
                x1 = args.rng.integers(2*(roi//3)-s,2*(roi//3)+s+1)
                y0 = args.rng.integers(1*(roi//3)-s,1*(roi//3)+s+1)
                y1 = args.rng.integers(2*(roi//3)-s,2*(roi//3)+s+1)
                ir.append(Rectangle(0, 0, x0, y0)) # 0
                ir.append(Rectangle(x0, 0, x1, y0)) # 1
                ir.append(Rectangle(x1, 0, roi, y0)) # 2
                ir.append(Rectangle(0, y0, x0, y1)) # 3
                ir.append(Rectangle(x0, y0, x1, y1)) # 4
                ir.append(Rectangle(x1, y0, roi, y1)) # 5
                ir.append(Rectangle(0, y1, x0, roi)) # 6
                ir.append(Rectangle(x0, y1, x1, roi)) # 7
                ir.append(Rectangle(x1, y1, roi, roi)) # 8
    
            # populate dimg[] with len(ir) images
            for r in ir:
                while True:
                    fn = random.choice(self.flist[self.origin:self.origin+args.nsamples])
                    #fn = random.choice(self.flist)
                    img = cv2.imread('{}/imagenet/train/{}'.format(args.imagenet,fn.rstrip()))
                    if img is not None:
                        break
                # center crop
                side1 = min(img.shape[0],img.shape[1])
                img = img[img.shape[0]//2-side1//2:img.shape[0]//2+side1//2,img.shape[1]//2-side1//2:img.shape[1]//2+side1//2]
                sx = r.xmax-r.xmin
                sy = r.ymax-r.ymin
                m2 = max(sx,sy)
                img = cv2.resize(img,dsize=(m2,m2),interpolation=cv2.INTER_CUBIC)
                dx = (m2-sx)//2
                dy = (m2-sy)//2
                dimg[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = img[dy:dy+sy,dx:dx+sx]
    
        if noise:
            d = args.rng.integers(0,256,d.shape,d.dtype)
        if show:
            cv2.imshow('ref', dimg[args.rng.integers(args.batch)])
        d = dimg/255.
        d = np.rollaxis(d,-1,1)
        d = d.astype(np.float32)
        
        dref = [cv2.resize(img,dsize=(224,224),interpolation=cv2.INTER_CUBIC) for img in dimg]
        dref = np.array(dref)
        dref = dref/255.
        dmean = self.transforms.transforms[-1].mean.numpy()
        dstd = self.transforms.transforms[-1].std.numpy()
        #print('dmean',dmean,'dstd',dstd)
        dref = np.subtract(dref,dmean)
        dref = np.divide(dref,dstd)
        dref = np.rollaxis(dref,-1,1)
        dref = dref.astype(np.float32)
        dref=torch.utils.data.default_convert(dref)
        dref = dref.to(args.device)
        l = self.ref(dref)[3].detach().cpu().numpy()
        return d,l

dataset = Batch(args,ref,transforms)

if args.train:
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
            d,l = dataset.generate_batch(args,noise=False)
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
        (x0,y0)=q.get()
        #print('y0',y0.shape,'min',np.min(y0),'max',np.max(y0),'mean',np.mean(y0),'std',np.std(y0))
        #print(np.histogram(y0))
        #lm = np.sign(np.mean(y0,axis=-3))
        #lm = np.mean(y0,axis=-3)
        #print('lm',lm.shape,lm)
        #exit()
        x=torch.utils.data.default_convert(x0)
        x = x.to(device)
        y=torch.utils.data.default_convert(y0)
        y = y.to(device)
        fmap = encoder(x)
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
            loss = torch.mean(1-cos_loss(fmap,y))
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
            loss = mse_loss(fmap,y)
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
            if args.sched=='grad':
                scheduler.step(gavg)
            if args.sched=='plat':
                scheduler.step(gavg)
            else:
                scheduler.step()
    
        if (args.shuffle>0) and (i%args.shuffle)==0:
            dataset.shuffle()
        #if (args.slide>0) and (gavg>args.slide):
        #if (args.slide>0) and (lavg<args.slide):
        if (args.slide>0) and (i%(int(args.slide)))==0:
            dataset.slide()

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
        s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} opt {:15} fmap {:12.6f} {:12.6f} ref {:12.6f} {:12.6f} ns {:9d} {:9d}'.format(
        i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,args.opt+'_'+args.loss+'_'+args.sched,torch.mean(fmap),torch.std(fmap),np.mean(y0),np.std(y0),args.nsamples,dataset.origin)
        print(s)
        with open(args.log, 'a') as f:
            print(s,file=f)
        
        if args.shuffle<0:
            saveinterval=1000
        else:
            saveinterval=args.shuffle
        if args.save is not None and (i%saveinterval)==0:
            torch.save(encoder.state_dict(), '{}'.format(args.save))
            encoder.save_pretrained('ie120nx_{}'.format(siliconperception.__version__))
    
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





exit()
