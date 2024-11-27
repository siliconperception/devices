import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from collections import OrderedDict

# PyTorch model for Silicon Perception IE120-R single chip image encoder
# www.siliconperception.com
class IE120R(
    nn.Module,
    PyTorchModelHubMixin, 
    repo_url="https://github.com/siliconperception/models",
    license="mit",
):
    def __init__(self,alt='medium'):
        super().__init__()
        if alt=='large':
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
            self.layer10 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU())
            self.layer11 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(96), nn.ReLU())
            self.layer12 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(96), nn.ReLU())
            self.layerf1  = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
            self.layer13 = nn.Sequential(nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(192), nn.ReLU())
            self.layer14 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU())
            self.layer15 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU())
            self.layerf2  = nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0)
            self.layer16 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(384), nn.ReLU())
            self.layer17 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU())
            self.layer18 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU())
            self.layerf3  = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
            self.layer19 = nn.Sequential(nn.Conv2d(384, 640, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(640), nn.ReLU())
            self.layer20 = nn.Sequential(nn.Conv2d(640, 640, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(640), nn.ReLU())
            self.layer21 = nn.Sequential(nn.Conv2d(640, 640, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(640), nn.ReLU())
            self.layerf4  = nn.Conv2d(640, 512, kernel_size=1, stride=1, padding=0)
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

# create a new PT model 1) 7x7 output only 2) merge batchnorm 3) quantize weights to FP8
# save the new PT model and write .memh files for the weights and bias
class IE120R_HW(nn.Module):
    def __init__(self,model):
        # merge BN with previous Conv2D, quantize weights to FP8, replace FP32 weights with quantized version
        super().__init__()
        self.merge_batchnorm(model.eval().to('cpu'))
        self.quantize()

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

    def fp8(self,w_fp32):
        # e4m3
        w_fp32 = np.maximum(np.abs(w_fp32),0.015625)*np.sign(w_fp32)
        w_fp32 = np.minimum(np.abs(w_fp32),448)*np.sign(w_fp32)
        w = np.frombuffer(w_fp32.tobytes(), dtype=np.uint32)
        s = np.bitwise_and(w,0x80000000)
        e = np.bitwise_and(w,0x7f800000)
        m = np.bitwise_and(w,0x00700000)
        e1 = e-(127<<23)+(7<<23)
        e1 = np.bitwise_and(e1,0x07800000)
        e = np.where(e==0,np.zeros(e.shape,dtype=np.uint32),e1)
        fp8 = np.bitwise_or(np.right_shift(s,24),np.bitwise_or(np.right_shift(e,20),np.right_shift(m,20))).astype(np.uint8)
        fp8 = fp8.reshape(w_fp32.shape)
        return fp8
    
    def fp32(self,x):
        # e4m3
        s = (x&0x80)>>7
        e = (x&0x78)>>3
        m = (x&0x07)>>0
        s = s.astype(np.uint32)
        e = e.astype(np.uint32)
        m = m.astype(np.uint32)
        w = (s<<31)|(e<<23)|(m<<20)
        w_fp32 = np.frombuffer(w.tobytes(), dtype=np.float32)
        w_fp32 = np.copy(w_fp32.reshape(x.shape))
        return w_fp32

    def merge_batchnorm(self,model):
        self.layer1 = nn.Sequential(self.fuse_conv_and_bn(model.layer1[0],model.layer1[1]), nn.ReLU())
        self.layer2 = nn.Sequential(self.fuse_conv_and_bn(model.layer2[0],model.layer2[1]), nn.ReLU())
        self.layer3 = nn.Sequential(self.fuse_conv_and_bn(model.layer3[0],model.layer3[1]), nn.ReLU())
        self.layer4 = nn.Sequential(self.fuse_conv_and_bn(model.layer4[0],model.layer4[1]), nn.ReLU())
        self.layer5 = nn.Sequential(self.fuse_conv_and_bn(model.layer5[0],model.layer5[1]), nn.ReLU())
        self.layer6 = nn.Sequential(self.fuse_conv_and_bn(model.layer6[0],model.layer6[1]), nn.ReLU())
        self.layer7 = nn.Sequential(self.fuse_conv_and_bn(model.layer7[0],model.layer7[1]), nn.ReLU())
        self.layer8 = nn.Sequential(self.fuse_conv_and_bn(model.layer8[0],model.layer8[1]), nn.ReLU())
        self.layer9 = nn.Sequential(self.fuse_conv_and_bn(model.layer9[0],model.layer9[1]), nn.ReLU())
        self.layer10 = nn.Sequential(self.fuse_conv_and_bn(model.layer10[0],model.layer10[1]), nn.ReLU())
        self.layer11 = nn.Sequential(self.fuse_conv_and_bn(model.layer11[0],model.layer11[1]), nn.ReLU())
        self.layer12 = nn.Sequential(self.fuse_conv_and_bn(model.layer12[0],model.layer12[1]), nn.ReLU())
        self.layer13 = nn.Sequential(self.fuse_conv_and_bn(model.layer13[0],model.layer13[1]), nn.ReLU())
        self.layer14 = nn.Sequential(self.fuse_conv_and_bn(model.layer14[0],model.layer14[1]), nn.ReLU())
        self.layer15 = nn.Sequential(self.fuse_conv_and_bn(model.layer15[0],model.layer15[1]), nn.ReLU())
        self.layer16 = nn.Sequential(self.fuse_conv_and_bn(model.layer16[0],model.layer16[1]), nn.ReLU())
        self.layer17 = nn.Sequential(self.fuse_conv_and_bn(model.layer17[0],model.layer17[1]), nn.ReLU())
        self.layer18 = nn.Sequential(self.fuse_conv_and_bn(model.layer18[0],model.layer18[1]), nn.ReLU())
        self.layer19 = nn.Sequential(self.fuse_conv_and_bn(model.layer19[0],model.layer19[1]), nn.ReLU())
        self.layer20 = nn.Sequential(self.fuse_conv_and_bn(model.layer20[0],model.layer20[1]), nn.ReLU())
        self.layer21 = nn.Sequential(self.fuse_conv_and_bn(model.layer21[0],model.layer21[1]), nn.ReLU())
        self.layerf4 = model.layerf4

    def quantize(self):
        self.weight1 = self.fp8(self.layer1[0].weight.detach().numpy())
        self.weight2 = self.fp8(self.layer2[0].weight.detach().numpy())
        self.weight3 = self.fp8(self.layer3[0].weight.detach().numpy())
        self.weight4 = self.fp8(self.layer4[0].weight.detach().numpy())
        self.weight5 = self.fp8(self.layer5[0].weight.detach().numpy())
        self.weight6 = self.fp8(self.layer6[0].weight.detach().numpy())
        self.weight7 = self.fp8(self.layer7[0].weight.detach().numpy())
        self.weight8 = self.fp8(self.layer8[0].weight.detach().numpy())
        self.weight9 = self.fp8(self.layer9[0].weight.detach().numpy())
        self.weight10 = self.fp8(self.layer10[0].weight.detach().numpy())
        self.weight11 = self.fp8(self.layer11[0].weight.detach().numpy())
        self.weight12 = self.fp8(self.layer12[0].weight.detach().numpy())
        self.weight13 = self.fp8(self.layer13[0].weight.detach().numpy())
        self.weight14 = self.fp8(self.layer14[0].weight.detach().numpy())
        self.weight15 = self.fp8(self.layer15[0].weight.detach().numpy())
        self.weight16 = self.fp8(self.layer16[0].weight.detach().numpy())
        self.weight17 = self.fp8(self.layer17[0].weight.detach().numpy())
        self.weight18 = self.fp8(self.layer18[0].weight.detach().numpy())
        self.weight19 = self.fp8(self.layer19[0].weight.detach().numpy())
        self.weight20 = self.fp8(self.layer20[0].weight.detach().numpy())
        self.weight21 = self.fp8(self.layer21[0].weight.detach().numpy())
        self.weightf4 = self.fp8(self.layerf4.weight.detach().numpy())
        self.layer1[0].weight.data = torch.from_numpy(self.fp32(self.weight1))
        self.layer2[0].weight.data = torch.from_numpy(self.fp32(self.weight2))
        self.layer3[0].weight.data = torch.from_numpy(self.fp32(self.weight3))
        self.layer4[0].weight.data = torch.from_numpy(self.fp32(self.weight4))
        self.layer5[0].weight.data = torch.from_numpy(self.fp32(self.weight5))
        self.layer6[0].weight.data = torch.from_numpy(self.fp32(self.weight6))
        self.layer7[0].weight.data = torch.from_numpy(self.fp32(self.weight7))
        self.layer8[0].weight.data = torch.from_numpy(self.fp32(self.weight8))
        self.layer9[0].weight.data = torch.from_numpy(self.fp32(self.weight9))
        self.layer10[0].weight.data = torch.from_numpy(self.fp32(self.weight10))
        self.layer11[0].weight.data = torch.from_numpy(self.fp32(self.weight11))
        self.layer12[0].weight.data = torch.from_numpy(self.fp32(self.weight12))
        self.layer13[0].weight.data = torch.from_numpy(self.fp32(self.weight13))
        self.layer14[0].weight.data = torch.from_numpy(self.fp32(self.weight14))
        self.layer15[0].weight.data = torch.from_numpy(self.fp32(self.weight15))
        self.layer16[0].weight.data = torch.from_numpy(self.fp32(self.weight16))
        self.layer17[0].weight.data = torch.from_numpy(self.fp32(self.weight17))
        self.layer18[0].weight.data = torch.from_numpy(self.fp32(self.weight18))
        self.layer19[0].weight.data = torch.from_numpy(self.fp32(self.weight19))
        self.layer20[0].weight.data = torch.from_numpy(self.fp32(self.weight20))
        self.layer21[0].weight.data = torch.from_numpy(self.fp32(self.weight21))
        self.layerf4.weight.data = torch.from_numpy(self.fp32(self.weightf4))
    
    def write_memh(self,fn_prefix='./'):
        print('layer1',self.weight1.shape,self.weight1.dtype, self.layer1[0].bias.cpu().detach().numpy().shape,self.layer1[0].bias.cpu().detach().numpy().dtype, self.out1.shape,self.out1.dtype)
        pass

    def forward(self, x):
        self.out1 = self.layer1(x) # save intermediate tensors for write_memh
        self.out2 = self.layer2(self.out1)
        self.out3 = self.layer3(self.out2)
        self.out4 = self.layer4(self.out3)
        self.out5 = self.layer5(self.out4)
        self.out6 = self.layer6(self.out5)
        self.out7 = self.layer7(self.out6)
        self.out8 = self.layer8(self.out7)
        self.out9 = self.layer9(self.out8)
        self.out10 = self.layer10(self.out9)
        self.out11 = self.layer11(self.out10)
        self.out12 = self.layer12(self.out11)
        self.out13 = self.layer13(self.out12)
        self.out14 = self.layer14(self.out13)
        self.out15 = self.layer15(self.out14)
        self.out16 = self.layer16(self.out15)
        self.out17 = self.layer17(self.out16)
        self.out18 = self.layer18(self.out17)
        self.out19 = self.layer19(self.out18)
        self.out20 = self.layer20(self.out19)
        self.out21 = self.layer21(self.out20)
        self.out = self.layerf4(self.out21)
        return self.out

