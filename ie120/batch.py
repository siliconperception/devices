from collections import namedtuple
import random
import numpy as np
import scipy
import cv2

class Batch:
    def __init__(self,args):
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

    def generate_batch(self,args):
        d = np.zeros([args.batch,args.roi,args.roi,3]).astype(np.uint8)
        l0 = np.zeros([args.batch,1000,3,3]).astype(float) # class probabilities 3x3 feature map
        l1 = np.zeros([args.batch,1000,2,2]).astype(float) # class probabilities 2x2 feature map
        l2 = np.zeros([args.batch,1000,1,1]).astype(float) # class probabilities 1x1 feature map
    
        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
        choices = []
        if args.x1:
            choices.extend(1*['1x1'])
        if args.x2:
            choices.extend(1*['2x2'])
        if args.x3:
            choices.extend(1*['3x3'])
    
        for i in range(args.batch):
            sc = random.choice(choices)
            ir=[] # image rectangles
            roi=args.roi
            if sc=='1x1':
                ir.append(Rectangle(0, 0, roi, roi))
            if sc=='2x2':
                s = (roi*args.resize)//2
                x0 = np.random.randint((roi//2)-s,(roi//2)+s+1)
                y0 = np.random.randint((roi//2)-s,(roi//2)+s+1)
                ir.append(Rectangle(0, 0, x0, y0))
                ir.append(Rectangle(x0, 0, roi, y0))
                ir.append(Rectangle(0, y0, x0, roi))
                ir.append(Rectangle(x0, y0, roi, roi))
            if sc=='3x3':
                s = (roi*args.resize)//3
                x0 = np.random.randint(1*(roi//3)-s,1*(roi//3)+s+1)
                x1 = np.random.randint(2*(roi//3)-s,2*(roi//3)+s+1)
                y0 = np.random.randint(1*(roi//3)-s,1*(roi//3)+s+1)
                y1 = np.random.randint(2*(roi//3)-s,2*(roi//3)+s+1)
                ir.append(Rectangle(0, 0, x0, y0)) # 0
                ir.append(Rectangle(x0, 0, x1, y0)) # 1
                ir.append(Rectangle(x1, 0, roi, y0)) # 2
                ir.append(Rectangle(0, y0, x0, y1)) # 3
                ir.append(Rectangle(x0, y0, x1, y1)) # 4
                ir.append(Rectangle(x1, y0, roi, y1)) # 5
                ir.append(Rectangle(0, y1, x0, roi)) # 6
                ir.append(Rectangle(x0, y1, x1, roi)) # 7
                ir.append(Rectangle(x1, y1, roi, roi)) # 8
    
            # populate d[] with len(ir) images
            il=[] # image labels corresponding to ir[]
            for r in ir:
                while True:
                    fn = random.choice(self.flist)
                    img = cv2.imread('{}/imagenet/train/{}'.format(args.imagenet,fn.rstrip()))
                    if img is not None:
                        break
                if args.centercrop:
                    side1 = min(img.shape[0],img.shape[1])
                    img = img[img.shape[0]//2-side1//2:img.shape[0]//2+side1//2,img.shape[1]//2-side1//2:img.shape[1]//2+side1//2]
                    sx = r.xmax-r.xmin
                    sy = r.ymax-r.ymin
                    m2 = max(sx,sy)
                    img = cv2.resize(img,dsize=(m2,m2),interpolation=cv2.INTER_CUBIC)
                    dx = (m2-sx)//2
                    dy = (m2-sy)//2
                    d[i,r.ymin:r.ymax,r.xmin:r.xmax,:] = img[dy:dy+sy,dx:dx+sx]
                il.append(self.synlabel[fn[0:9]]-1)
    
            # for each rf, populate l[] with len(ir) [0,1] probabilities based on overlap between rf and ir rectangles
            if sc=='1x1':
                l2[i,il[0],0,0] = 1
            if sc=='2x2':
                l1[i,il[0],0,0] = 1
                l1[i,il[1],0,1] = 1
                l1[i,il[2],1,0] = 1
                l1[i,il[3],1,1] = 1
            if sc=='3x3':
                l0[i,il[0],0,0] = 1
                l0[i,il[1],0,1] = 1
                l0[i,il[2],0,2] = 1
                l0[i,il[3],1,0] = 1
                l0[i,il[4],1,1] = 1
                l0[i,il[5],1,2] = 1
                l0[i,il[6],2,0] = 1
                l0[i,il[7],2,1] = 1
                l0[i,il[8],2,2] = 1
    
        d = d.astype(np.float32)/255.
        d = np.rollaxis(d,-1,1)
        l=[]
        l.append(l2[:,:,0,0])
        l.append(l1[:,:,0,0])
        l.append(l1[:,:,0,1])
        l.append(l1[:,:,1,0])
        l.append(l1[:,:,1,1])
        l.append(l0[:,:,0,0])
        l.append(l0[:,:,0,1])
        l.append(l0[:,:,0,2])
        l.append(l0[:,:,1,0])
        l.append(l0[:,:,1,1])
        l.append(l0[:,:,1,2])
        l.append(l0[:,:,2,0])
        l.append(l0[:,:,2,1])
        l.append(l0[:,:,2,2])
        l = np.stack(l,axis=-1)
        return d,l
