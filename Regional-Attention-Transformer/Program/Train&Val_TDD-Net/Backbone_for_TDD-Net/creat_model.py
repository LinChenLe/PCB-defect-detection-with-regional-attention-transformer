from torch import nn
import torch
import pdb
from math import sqrt
from torch.nn.parameter import Parameter
import copy
import math
import torch.nn.functional as F
import torchvision
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# =============================================================================
# segmentation
num_channel = 256
drop_rate = 0.5
class creat_backbone_main_model(nn.Module):
    def __init__(self,backbone_model):
        super().__init__()
        self.backbone_model = backbone_model
        self._reset_parameters()
    def _reset_parameters(self):
        for n,p in self.named_parameters():
            print(n)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self,tested):
        img = tested
        result = self.backbone_model(img)
        
        return result
        
class norm2D(nn.Module):
    def __init__(self,num_channel):
        super().__init__()
        self.gamma = Parameter(torch.ones(1,num_channel,1,1),requires_grad=True)#[None,:,None,None]
        self.beta = Parameter(torch.zeros(1,num_channel,1,1),requires_grad=True)#[None,:,None,None]
    def forward(self,inputs):
        mean = torch.mean(inputs,dim=(-1,-2),keepdim=True)
        std = torch.std(inputs,dim=(-1,-2),keepdim=True)+1e-05
        return (((inputs-mean)/std)*self.gamma)+self.beta
    
class norm1D(nn.Module):
    def __init__(self, gamma=1, beta=0):
        super().__init__()
        self.gamma = Parameter(torch.full([1],gamma,dtype=torch.float32),requires_grad=True)
        self.beta = Parameter(torch.full([1],beta,dtype=torch.float32),requires_grad=True)
    def forward(self,inputs):
        mean = torch.mean(inputs,dim=(-1),keepdim=True)
        std = torch.std(inputs,dim=(-1),keepdim=True)+1e-05
        inputs = (((inputs-mean)/std)*self.gamma)+self.beta

        return inputs


'''ResNet-18 Image classfication for cifar-10 with PyTorch 

Author 'Sun-qian'.

'''

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            norm2D(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            norm2D(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride),
                norm2D(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=[7,7], stride=2, padding=3),
            norm2D(64),
            nn.ReLU(),
        )
        self.max_pool2d = nn.MaxPool2d([3,3],stride=2,padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pool2d(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)

        out = out.view(out.size(0), -1)
        
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock,6)

def build_backbone_model(args):
    backbone_model = ResNet18().to(args.device)
    # backbone_model = resnet50()
    backbone_model = creat_backbone_main_model(backbone_model).to(args.device)
    return backbone_model


# =============================================================================
if __name__ == '__main__':
    model = build_backbone_model('gray', 100, 2048, 256, 6)
    
    tensor = torch.rand([2,1,512,512],dtype=torch.float32)
    
    result = model(tensor)
    print(result['bbox'].requires_grad,result['category'].requires_grad)
