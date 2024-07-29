import torch
import torch.nn as nn
class Imagenet(nn.Module):
    def __init__(self, encoder, alt='alt1'):
        super(Imagenet, self).__init__()
        self.encoder = encoder
        self.alt = alt
        if alt=='alt1':
            self.head0 = nn.Conv2d(512, 1000, kernel_size=(1,1), stride=1) # 3x3 feature map
            self.head1 = nn.Conv2d(512, 1000, kernel_size=(2,2), stride=1) # 2x2 feature map
            self.head2 = nn.Conv2d(512, 1000, kernel_size=(3,3), stride=1) # 1x1 feature map
            #heads=[]
            #heads.append(nn.ModuleList([nn.Conv2d(512, 1000, kernel_size=(1,1), stride=1)])) # 3x3 feature map
            #heads.append(nn.ModuleList([nn.Conv2d(512, 1000, kernel_size=(2,2), stride=1)])) # 2x2 feature map
            #heads.append(nn.ModuleList([nn.Conv2d(512, 1000, kernel_size=(3,3), stride=1)])) # 1x1 feature map

#            projection=512
#            self.projection = projection
#            for i in range(1+4+9):
#                head=[]
#                head.append(nn.Conv2d(512, 1000, kernel_size=(2,2), stride=1))
#                #head.append(nn.Conv2d(512, 1000, kernel_size=(3,3), stride=1))
#                #head.append(nn.Conv2d(512, projection, kernel_size=(2,2), stride=1))
#                #head.append(nn.SELU())
#                #head.append(nn.Conv2d(projection, projection, kernel_size=1, stride=1))
#                #head.append(nn.SELU())
#                #head.append(nn.Conv2d(projection, projection, kernel_size=1, stride=1))
#                #head.append(nn.SELU())
#                #head.append(nn.Conv2d(projection, projection, kernel_size=1, stride=1))
#                #head.append(nn.SELU())
#                #head.append(nn.Conv2d(projection, projection, kernel_size=1, stride=1))
#                #head.append(nn.SELU())
#                #head.append(nn.Conv2d(projection, 1000, kernel_size=1, stride=1))
#                heads.append(nn.ModuleList(head))

#            self.heads = nn.ModuleList(heads)

            #self.task = nn.Conv2d(512, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layer1 = nn.Conv2d(512, self.projection, kernel_size=(2,2), stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layer1b = nn.SELU()
            #self.layer2 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layer2b = nn.SELU()
            #self.layer3 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layer3b = nn.SELU()
            #self.layer4 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layer4b = nn.SELU()
            #self.layer5 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layer5b = nn.SELU()
            #self.layerp0 = nn.Conv2d(self.projection, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layerp1 = nn.Conv2d(self.projection, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layerp2 = nn.Conv2d(self.projection, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layerp3 = nn.Conv2d(self.projection, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layer0 = nn.ReLU()
            #self.layer1 = nn.Flatten()
            #self.layer2 = nn.Linear(2048,self.projection)
            #self.layer2b = nn.ReLU()
            #self.layer3 = nn.Conv2d(512+self.projection, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            
        if alt=='alt2':
            self.projection=1000
            self.layer1 = nn.Conv2d(512, self.projection, kernel_size=(3,3), stride=1) # linearly project [3,3,512] features to [1000] classes
            self.layer1b = nn.SELU()
            self.layer2 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.layer2b = nn.SELU()
            #self.layer3 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layer3b = nn.SELU()
            #self.layer4 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layer4b = nn.SELU()
            #self.layer5 = nn.Conv2d(self.projection, self.projection, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #self.layer5b = nn.SELU()
            self.layerl = nn.Conv2d(self.projection, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes

            #self.layerl = nn.Conv2d(512, 1000, kernel_size=(2,2), stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            #decoder=1000
            #self.layer1 = nn.Flatten()
            #self.layer2 = nn.Linear(2048,1000)
            #self.layer2 = nn.Linear(2048,decoder*1)
            #self.layer2a = nn.BatchNorm1d(decoder*1)
            #self.layer2b = nn.ReLU()
            #self.layer3 = nn.Linear(decoder*1,decoder*1)
            #self.layer3a = nn.BatchNorm1d(decoder*1)
            #self.layer3b = nn.ReLU()
            #self.layer4 = nn.Linear(decoder*1,decoder*1)
            #self.layer4a = nn.BatchNorm1d(decoder*1)
            #self.layer4b = nn.ReLU()
            #self.layer5 = nn.Linear(decoder*1,decoder*1)
            #self.layer5a = nn.BatchNorm1d(decoder*1)
            #self.layer5b = nn.ReLU()
            #self.layer6 = nn.Linear(decoder*1,decoder*1)
            #self.layer6a = nn.BatchNorm1d(decoder*1)
            #self.layer6b = nn.ReLU()
            #self.layer7 = nn.Linear(decoder*1,1000)
            #self.layer3 = nn.Linear(1000,1000)
            #self.layer3b = nn.ReLU()
            #self.layer4 = nn.Linear(1000,1000)
            #self.layer4b = nn.ReLU()
            #self.layer5 = nn.Linear(1000,1000)
            #self.layer5b = nn.SELU()
            #self.layer6 = nn.Linear(1000,1000)
            #self.layer6b = nn.SELU()
            #self.layer7 = nn.Linear(1000,1000)
            #self.layer5a = nn.BatchNorm1d(1000)
            #self.layer5b = nn.ReLU()
            #self.layer6 = nn.Linear(1000,1000)
        if alt=='alt3':
            self.classify = nn.Conv2d(512, 1000, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.xscale = nn.Conv2d(512, 700, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.yscale = nn.Conv2d(512, 700, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes
            self.xyflip = nn.Conv2d(512, 4, kernel_size=1, stride=1) # linearly project [2,2,512] features to [2,2,1000] classes

    def forward(self, x):
        fmap = self.encoder(x)
        if self.alt=='alt1':
            y0 = self.head0(fmap)
            y1 = self.head1(fmap)
            y2 = self.head2(fmap)
            y=[]
            y.append(y2[:,:,0,0])
            y.append(y1[:,:,0,0])
            y.append(y1[:,:,0,1])
            y.append(y1[:,:,1,0])
            y.append(y1[:,:,1,1])
            y.append(y0[:,:,0,0])
            y.append(y0[:,:,0,1])
            y.append(y0[:,:,0,2])
            y.append(y0[:,:,1,0])
            y.append(y0[:,:,1,1])
            y.append(y0[:,:,1,2])
            y.append(y0[:,:,2,0])
            y.append(y0[:,:,2,1])
            y.append(y0[:,:,2,2])
            y = torch.stack(y,dim=-1)
            #print('y',y.shape)

#            yhead=[]
#            for head in self.heads:
#                y = fmap
#                for h in head:
#                    #print('h',h,type(h),dir(h))
#                    y = h(y)
#                yhead.append(y)
#                #print('yhead',y.shape)
#            y = torch.stack(yhead,dim=-1)
            #y = torch.reshape(y,[-1,1000,2,2])

            #y = self.layer1b(self.layer1(fmap))
            #y = self.layer2b(self.layer2(y))
            #y = self.layer3b(self.layer3(y))
            #y = self.layer4b(self.layer4(y))
            #y = self.layer5b(self.layer5(y))
            #y0 = self.layerp0(y)
            #y1 = self.layerp1(y)
            #y2 = self.layerp2(y)
            #y3 = self.layerp3(y)
            #y = torch.stack([y0,y1,y2,y3],dim=-1)
            #y = torch.reshape(y,[-1,1000,2,2])

            #y = self.task(fmap)
            #y0 = self.layer0(fmap) # RELU
            #y1 = self.layer1(y0)
            #y2a = self.layer2(y1)
            #y2 = self.layer2b(y2a) # RELU
            #y3 = torch.dstack([y2,y2,y2,y2])
            #y4 = torch.reshape(y3,[-1,self.projection,2,2])
            #y5 = torch.cat([y0,y4],dim=1)
            #y = self.layer3(y5)
            #y3 = y = torch.cat([y0,y1,y2,y3],dim=1)
            #y0 = self.layer0(fmap)
            # print('fmap',fmap.shape,'y1',y1.shape,'y2',y2.shape,'y3',y3.shape,'y4',y4.shape,'y5',y5.shape)
            #exit()
            return y
        if self.alt=='alt2':
            #y = self.layerl(fmap)
            y = self.layer1b(self.layer1(fmap))
            y = self.layer2b(self.layer2(y))
            #y = self.layer3b(self.layer3(y))
            #y = self.layer4b(self.layer4(y))
            #y = self.layer5b(self.layer5(y))
            y = self.layerl(y)
            return y
        if self.alt=='alt3':
            return self.classify(fmap), self.xscale(fmap), self.yscale(fmap), self.xyflip(fmap)


