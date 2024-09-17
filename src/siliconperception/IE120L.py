import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

# PyTorch model for Silicon Perception IE120-L single chip image encoder, which executes the model at 200 frames/s with 150us latency
# www.siliconperception.com
class IE120L(
    nn.Module,
    PyTorchModelHubMixin, 
    repo_url="https://github.com/siliconperception/models",
    license="mit",
):
    def __init__(self):
        super().__init__()
        self.layer1  = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2), nn.BatchNorm2d(16), nn.ReLU())
        self.layer2  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1), nn.BatchNorm2d(16), nn.ReLU())
        self.layer3  = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1), nn.BatchNorm2d(16), nn.ReLU())
        self.layer4  = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.BatchNorm2d(32), nn.ReLU())
        self.layer5  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer6  = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer7  = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU())
        self.layer8  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer9  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer10 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2), nn.BatchNorm2d(128), nn.ReLU())
        self.layer11 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer12 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer13 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.BatchNorm2d(256), nn.ReLU())
        self.layer14 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer15 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer16 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2), nn.BatchNorm2d(512), nn.ReLU())
        self.layer17 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1), nn.BatchNorm2d(512), nn.ReLU())
        self.layer18 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1), nn.BatchNorm2d(512), nn.ReLU())

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
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        return out

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
        enc = nn.Sequential(collections.OrderedDict([
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

