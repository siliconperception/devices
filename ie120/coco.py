from pycocotools.coco import COCO
import numpy as np
import torch
import torch.nn as nn
import torchinfo
import cv2
import argparse
from pprint import pprint
import random
import ie120
import threading
import queue
import subprocess
import os
import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', help='pytorch execution device',default='cpu')
#parser.add_argument('--null', help='null probability threshold',default=0.5, type=float)
parser.add_argument('--decoder', help='decoder model name',default=None)
parser.add_argument('--nbatch', help='total training batches',default=1000000000000, type=int)
parser.add_argument('--save', help='trained model name',default='decoder.pt')
parser.add_argument('--lr', help='initial learning rate',default=0.1, type=float)
parser.add_argument('--weight_decay', help='L2 penalty',default=0.0, type=float)
parser.add_argument('--nesterov', help='SGD param',default=False, action='store_true')
parser.add_argument('--momentum', help='SGD param',default=0.0, type=float)
parser.add_argument('--factor', help='LR schedule param',default=0.7, type=float)
parser.add_argument('--step', help='LR scheduler batches per step',default=1000000000, type=int)
parser.add_argument('--avg', help='moving average window for lr and grad',default=100, type=int)
parser.add_argument('--workers', help='number of threads for batch generation',default=20, type=int)
parser.add_argument('--opt', help='optimizer type',default='sgd')
parser.add_argument('--sched', help='LR scheduler type',default=None)
parser.add_argument('--batch', help='batch size',default=1, type=int)
parser.add_argument('--train', help='train encoder-decoder model',default=False, action='store_true')
parser.add_argument('--test', help='test encoder-decoder model',default=False, action='store_true')
parser.add_argument('--ann', help='annotation json file',default='../coco/annotations/instances_train2017.json')
parser.add_argument('--img', help='image directory',default='../coco/train2017')
parser.add_argument('--encoder', help='encoder model name',default='ie120-sgd-scratch.pt')
parser.add_argument('--freeze', help='freeze encoder weights',default=False, action='store_true')
parser.add_argument('--alt', help='encoder model alt type',default='alt1')
parser.add_argument('--roi', help='input image x/y size',default=768, type=int)
parser.add_argument('--debug', help='verbose',default=False, action='store_true')
parser.add_argument('--log', help='log file name',default=None)
parser.add_argument('--show', help='display batches',default=False, action='store_true')
args = parser.parse_args()
if args.log is None:
    args.date = subprocess.check_output(['/usr/bin/date', '+%Y.%m.%d-%H.%M.%S'])
    args.date = args.date.decode("utf-8")
    args.date = args.date.rstrip()
    if not os.path.exists('log'):
        os.makedirs('log')
    args.log = 'log/log.{}'.format(args.date)
print(args)
with open(args.log, 'a') as f:
    print('ARGS',args,file=f)

encoder = ie120.IE120(args.alt)
print('image encoder model initialized')

encoder.load_state_dict(torch.load('{}'.format(args.encoder)))
print('image encoder model state_dict loaded')

if args.freeze:
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval() # eval mode
    print('image encoder model frozen')

else:
    encoder.train() # train mode

class Decoder(nn.Module):
    def __init__(self, encoder, alt='alt1'):
        super(Decoder, self).__init__()
        self.encoder = encoder
        self.alt = alt
        if alt=='alt1':
            c0=2000
            self.layer0d1  = nn.Sequential(nn.Conv2d(512, c0, kernel_size=1, stride=1), nn.BatchNorm2d(c0), nn.ReLU())
            self.layer0d2  = nn.Sequential(nn.Conv2d(c0, c0, kernel_size=1, stride=1), nn.BatchNorm2d(c0), nn.ReLU())
            self.layer0d3  = nn.Sequential(nn.Conv2d(c0, c0, kernel_size=1, stride=1), nn.BatchNorm2d(c0), nn.ReLU())
            self.layer0d4  = nn.Sequential(nn.Conv2d(c0, c0, kernel_size=1, stride=1), nn.BatchNorm2d(c0), nn.ReLU())
            self.layer0d5  = nn.Sequential(nn.Conv2d(c0, c0, kernel_size=1, stride=1), nn.BatchNorm2d(c0), nn.ReLU())
            self.layer0d6  = nn.Sequential(nn.Conv2d(c0, c0, kernel_size=1, stride=1), nn.BatchNorm2d(c0), nn.ReLU())
            self.layer0d7  = nn.Sequential(nn.Conv2d(c0, c0, kernel_size=1, stride=1), nn.BatchNorm2d(c0), nn.ReLU())
            c1=64
            self.layer1u = nn.Upsample(scale_factor=4, mode='nearest')
            self.layer1c1  = nn.Sequential(nn.Conv2d(c0, c1, kernel_size=3, stride=1), nn.BatchNorm2d(c1), nn.ReLU())
            self.layer1c2  = nn.Sequential(nn.Conv2d(c1, c1, kernel_size=3, stride=1), nn.BatchNorm2d(c1), nn.ReLU())
            self.layer1c3  = nn.Sequential(nn.Conv2d(c1, c1, kernel_size=3, stride=1), nn.BatchNorm2d(c1), nn.ReLU())
            c2=32
            self.layer2u = nn.Upsample(scale_factor=4, mode='nearest')
            self.layer2c1  = nn.Sequential(nn.Conv2d(c1, c2, kernel_size=3, stride=1), nn.BatchNorm2d(c2), nn.ReLU())
            self.layer2c2  = nn.Sequential(nn.Conv2d(c2, c2, kernel_size=3, stride=1), nn.BatchNorm2d(c2), nn.ReLU())
            self.layer2c3  = nn.Sequential(nn.Conv2d(c2, c2, kernel_size=3, stride=1), nn.BatchNorm2d(c2), nn.ReLU())
            c3=16
            self.layer3u = nn.Upsample(scale_factor=4, mode='nearest')
            self.layer3c1  = nn.Sequential(nn.Conv2d(c2, c3, kernel_size=3, stride=1), nn.BatchNorm2d(c3), nn.ReLU())
            self.layer3c2  = nn.Sequential(nn.Conv2d(c3, c3, kernel_size=3, stride=1), nn.BatchNorm2d(c3), nn.ReLU())
            self.layer3c3  = nn.Sequential(nn.Conv2d(c3, c3, kernel_size=3, stride=1), nn.BatchNorm2d(c3), nn.ReLU())
            c4=8
            self.layer4u = nn.Upsample(scale_factor=4, mode='nearest')
            self.layer4c1  = nn.Sequential(nn.Conv2d(c3, c4, kernel_size=3, stride=1), nn.BatchNorm2d(c4), nn.ReLU())
            self.layer4c2  = nn.Sequential(nn.Conv2d(c4, c4, kernel_size=3, stride=1), nn.BatchNorm2d(c4), nn.ReLU())
            self.layer4c3  = nn.Sequential(nn.Conv2d(c4, c4, kernel_size=3, stride=1), nn.BatchNorm2d(c4), nn.ReLU())
            c5=4
            self.layer5u = nn.Upsample(scale_factor=3, mode='nearest')
            self.layer5c1  = nn.Sequential(nn.Conv2d(c4, c5, kernel_size=3, stride=1), nn.BatchNorm2d(c5), nn.ReLU())
            self.layer5c2  = nn.Sequential(nn.Conv2d(c5, c5, kernel_size=3, stride=1), nn.BatchNorm2d(c5), nn.ReLU())
            self.layer5c3  = nn.Sequential(nn.Conv2d(c5, c5, kernel_size=3, stride=1), nn.BatchNorm2d(c5), nn.ReLU())
            self.layerl = nn.Conv2d(c5, 1+80, kernel_size=1, stride=1) # linear projection to 80 coco classes + null

    def forward(self, x):
        fmap = self.encoder(x)
        y = self.layer0d7(self.layer0d6(self.layer0d5(self.layer0d4(self.layer0d3(self.layer0d2(self.layer0d1(fmap)))))))
        y = self.layer1c3(self.layer1c2(self.layer1c1(self.layer1u(y))))
        y = self.layer2c3(self.layer2c2(self.layer2c1(self.layer2u(y))))
        y = self.layer3c3(self.layer3c2(self.layer3c1(self.layer3u(y))))
        y = self.layer4c3(self.layer4c2(self.layer4c1(self.layer4u(y))))
        y = self.layer5c3(self.layer5c2(self.layer5c1(self.layer5u(y))))
        y = self.layerl(y)
        return y

if args.decoder is None:
    model = Decoder(encoder,alt=args.alt)
else:
    model = torch.load('{}'.format(args.decoder))
torchinfo.summary(model, input_size=(1, 3, args.roi, args.roi))
device = torch.device(args.device)
model = model.to(device)

# BATCH CLASS USING COCO DATASET
class Batch:
    def __init__(self,args):
        self.coco=COCO(args.ann)
        self.img_ids = list(self.coco.imgs.keys())
        #self.img_ids = self.img_ids[0:20] # SMOKE TEST
        print('coco image_ids loaded',len(self.img_ids))
        self.cat_ids = dict(zip(self.coco.getCatIds(),range(1,1+80))) # map cat_id to label with null
        print('coco cat_ids loaded',len(self.cat_ids))
        cats = self.coco.loadCats(self.cat_ids)
        self.cat_names = ['null']
        self.cat_names.extend([cat["name"] for cat in cats])
        print('coco cat_names loaded',len(self.cat_names))

    def generate_batch(self,args):
        d = np.zeros([args.batch,args.roi,args.roi,3]).astype(np.float32)
        #l = np.zeros([args.batch,80,args.roi,args.roi]).astype(np.float32) # per pixel class probabilities
        l = np.zeros([args.batch,args.roi,args.roi]).astype(int) # per pixel class label
        for i in range(args.batch):
            while True:
                img_id = random.choice(self.img_ids)
                imgs = self.coco.imgs[img_id]
                img = cv2.imread('{}/{}'.format(args.img,imgs['file_name']))
                side = min(img.shape[0],img.shape[1])
                img = img[img.shape[0]//2-side//2:img.shape[0]//2+side//2,img.shape[1]//2-side//2:img.shape[1]//2+side//2]
                img = cv2.resize(img,dsize=(768,768),interpolation=cv2.INTER_CUBIC)
                if args.debug:
                    cv2.imshow('img', img)
                    if cv2.waitKey(0)==120: # 'x'
                        exit()
                d[i] = img.astype(np.float32)/255.
    
                ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
                anns = self.coco.loadAnns(ann_ids)
                random.shuffle(anns)
                for ann in anns:
                    cat_id = self.cat_ids[ann['category_id']]
                    mask = self.coco.annToMask(ann)
                    mask = mask[mask.shape[0]//2-side//2:mask.shape[0]//2+side//2,mask.shape[1]//2-side//2:mask.shape[1]//2+side//2]
                    mask *= 255
                    mask = cv2.resize(mask,dsize=(768,768),interpolation=cv2.INTER_CUBIC)
                    if args.debug:
                        cv2.imshow('label', mask)
                        if cv2.waitKey(0)==120: # 'x'
                            exit()
                    mask = np.divide(mask,255.)
                    mask = np.round(mask).astype(bool)
                    np.putmask(l[i],mask,cat_id)
                    #l[i,cat_id,:,:] = mask
                if np.any(l[i]!=0): # skip examples with no labels
                    break

        d = np.rollaxis(d,-1,1)
        #l = np.clip(l,0,1)
        return d,l

# color map for 80 coco categories
cmap=[]
cmap.append([0,0,0])
red = 0
grn = 100
blu = 200
for k in range(80):
    cmap.append([50+red,50+grn,50+blu])
    red = (red+70)%200
    grn = (grn+130)%200
    blu = (blu+170)%200
cmap = np.array(cmap,dtype=np.uint8)

if args.train:
    model.train()
    dataset = Batch(args)
    # TRAINING LOOP --save decoder.pt
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduce=None) # we will use class indices for softmax loss
    #criterion = nn.CrossEntropyLoss(reduce=None)
    #criterion = nn.BCEWithLogitsLoss()
    if args.opt=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.opt=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    if args.opt=='radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay,decoupled_weight_decay=True)
    if args.opt=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov,dampening=0.0)
    if args.sched=='plat':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=args.factor,min_lr=0.0001)
    
    def worker(stop, q, args, dataset):
        while not stop.is_set():
            (d,l) = dataset.generate_batch(args)
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
    garr=[]
    while i<1+args.nbatch:
        (x0,y0)=q.get()
        #print('y0',np.amin(y0),np.amax(y0))
        x=torch.utils.data.default_convert(x0)
        x = x.to(device)
        y=torch.utils.data.default_convert(y0)
        y = y.to(device)
         
        logits = model(x)
        loss = criterion(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        larr.append(loss.item())
        lavg = np.mean(larr[-args.avg:])
        if args.sched is not None and (i%args.step)==0:
            scheduler.step(lavg)
    
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
        s = 'BATCH {:12d} wall {} lr {:12.10f} wd {:12.10f} batch {:6d} loss {:12.6f} {:12.6f} grad {:12.6f} {:12.6f} opt {:15} mom {:12.10f} nest {}'.format(
            i,datetime.datetime.now(),lr,args.weight_decay,args.batch,loss.item(),lavg,total_norm,gavg,args.opt,mom,args.nesterov)
        print(s)
        with open(args.log, 'a') as f:
            print(s,file=f)
        
        if args.save is not None and (i%1000)==0:
            torch.save(model, '{}'.format(args.save))
    
        if args.show and (i%20)==0:
            img = x0[0]*255
            img = img.astype(np.uint8)
            img = np.swapaxes(img,0,-1)
            img = np.swapaxes(img,0,1)
            #cv2.imshow('coco', img)
            #print('cmap',cmap.shape,cmap)
            #print('y0',[k for k in y0[0]])
            gt = np.array([cmap[k] for k in y0[0]],dtype=np.uint8).reshape(img.shape)
            #print('y0',y0[0].shape, y0[0].dtype)
            #gt = cv2.applyColorMap(y0[0].astype(np.uint8), cv2.COLORMAP_HSV)

            #cv2.imshow('gt', gt)
            logits = logits[0]
            prob = torch.nn.functional.softmax(logits,dim=-3)
            #prob = prob.cpu().detach().numpy()
            #pred = np.argmax(prob,axis=-3)
            #mask = np.greater(prob[0],args.null) # null probability
            #pred = np.argmax(prob[1:,:,:],axis=-3)
            #np.putmask(pred,mask,0)
            pred = torch.argmax(prob,dim=-3)
            pred = pred.cpu().detach().numpy()
            #print('pred',pred.shape,pred)
            pimg = np.array([cmap[k] for k in pred],dtype=np.uint8).reshape(img.shape)
            #cv2.imshow('pred', pimg)
            cv2.imshow('coco', np.hstack([pimg,img,gt]))
            u,counts = np.unique(pred,return_counts=True)
            s = 'PRED wall {} cat_name '.format(datetime.datetime.now())
            for k in range(len(u)):
                s = s+dataset.cat_names[u[k]]+' '+str(counts[k])+','
            s = s[:-1]
            print(s)

            u,counts = np.unique(y0[0],return_counts=True)
            s = 'GT   wall {} cat_name '.format(datetime.datetime.now())
            for k in range(len(u)):
                s = s+dataset.cat_names[u[k]]+' '+str(counts[k])+','
            s = s[:-1]
            print(s)

            #cv2.waitKey(0)
            #exit(0)
#            prob = 1/(1+(np.exp((-logits)))) # sigmoid
#            #logits = np.exp(logits)
#            #logits = np.clip(logits,0,1)
#            mask = np.zeros_like(img)
#            for j in range(80):
#                if np.sum(prob[j])>0:
#                    for k in range(3):
#                        mask[:,:,k] = np.clip(mask[:,:,k]+np.round(prob[j]).astype(np.uint8)*np.random.randint(50,250),0,255)
#            cv2.imshow('mask', mask)
#            gt = np.zeros_like(img)
#            for j in range(80):
#                if np.sum(y0[0,j])>0:
#                    for k in range(3):
#                        gt[:,:,k] += np.round(y0[0,j]).astype(np.uint8)*np.random.randint(50,250)

        if args.show:
            if cv2.waitKey(1)==120: # 'x'
                exit()
    
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

if args.test:
    model.eval()
    dataset = Batch(args)
    while True:
        (d0,l) = dataset.generate_batch(args)
        d=torch.utils.data.default_convert(d0)
        d = d.to(device)
        logits = model(d)
        prob = torch.nn.functional.softmax(logits,dim=-3)
        pred = torch.argmax(prob,dim=-3)
        pred = pred.cpu().detach().numpy()
        gallery=[]
        for i in range(args.batch):
            img = d0[i]*255
            img = img.astype(np.uint8)
            img = np.swapaxes(img,0,-1)
            img = np.swapaxes(img,0,1)
            gt = np.array([cmap[k] for k in l[i]],dtype=np.uint8).reshape(img.shape)
            pimg = np.array([cmap[k] for k in pred[i]],dtype=np.uint8).reshape(img.shape)
            gallery.append(np.hstack([pimg,img,gt]))
            u,counts = np.unique(pred,return_counts=True)
            s = 'PRED wall {} cat_name '.format(datetime.datetime.now())
            for k in range(len(u)):
                s = s+dataset.cat_names[u[k]]+' '+str(counts[k])+','
            s = s[:-1]
            print(s)
    
            u,counts = np.unique(l[0],return_counts=True)
            s = 'GT   wall {} cat_name '.format(datetime.datetime.now())
            for k in range(len(u)):
                s = s+dataset.cat_names[u[k]]+' '+str(counts[k])+','
            s = s[:-1]
            print(s)

        img = np.vstack(gallery)
        print('img',img.shape)
        img = cv2.resize(img,dsize=(768,768),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('coco_test', img)
        if cv2.waitKey(0)==120: # 'x'
            exit()
    


exit()














dataset = Batch(args)
d,l = dataset.generate_batch(args)
print('d',d.shape,'l',l.shape)
print(np.sum(l))
exit()

exit()
coco=COCO(args.ann)

cat_ids = coco.getCatIds()
print(f"Number of Unique Categories: {len(cat_ids)}")
print("Category IDs:")
print(cat_ids)  # The IDs are not necessarily consecutive.

cats = coco.loadCats(cat_ids)
cat_names = [cat["name"] for cat in cats]
print("Categories Names:")
print(cat_names)

img_ids = list(coco.imgs.keys())
colors=[(200,0,0),(0,200,0),(0,0,200),(200,200,200),(100,100,100),(200,200,0),(0,200,200),(200,0,200)]
for j in range(100):
    img_id = random.choice(img_ids)
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    imgs = coco.imgs[img_id]
    img = cv2.imread('{}/{}'.format(args.img,imgs['file_name']))
    mask = np.zeros_like(img)
    for ann in anns:
        cat_id = ann['category_id']
        bbx0 = int(ann['bbox'][0])
        bby0 = int(ann['bbox'][1])
        bbx1 = int(ann['bbox'][0]+ann['bbox'][2])
        bby1 = int(ann['bbox'][1]+ann['bbox'][3])
        cats = coco.loadCats([cat_id])[0]
        #img = cv2.rectangle(img, (bbx0,bby0), (bbx1,bby1), (255,0,0), 1)
        #img = cv2.putText(img, cats['name'], (bbx0,bby0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
        color=random.choice(colors)
        m = np.full(img.shape,color,dtype=np.uint8)
        m0 = coco.annToMask(ann)
        m = np.multiply(m,np.stack([m0,m0,m0],axis=-1))
        mask = np.add(mask,m)
    cv2.imshow('coco', img)
    cv2.imshow('mask', mask)
    k=cv2.waitKey(0)
    if k==120: # 'x'
        break



print(f"Annotations for Image ID {img_id}:")
#print(anns)
print(len(anns))
exit()




for key in coco.anns.keys():
    pprint(coco.anns[key])
    idx = coco.anns[key]['image_id']
    break

pprint(coco.imgs[idx])
#for key in coco.imgs.keys():
#    pprint(coco.imgs[key])
#    print('key',key)
#    break
# Get all the annotations for the specified image.
ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
anns = coco_annotation.loadAnns(ann_ids)
print(f"Annotations for Image ID {img_id}:")
print(anns)
