import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch.model.util import SequentialWithIntermediates

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def forward_intermediates(self,x):
        outputs=[]
        out=self.conv1(x)
        outputs.append(out)
        out = F.relu(self.bn1(out))
        outputs.append(out)

        out=self.conv2(out)
        outputs.append(out)
        out = self.bn2(out)

        shortcut=self.shortcut(x)
        outputs.append(shortcut)
        out += self.shortcut(x)
        outputs.append(out)
        out = F.relu(out)
        outputs.append(out)
        return out,outputs
    def n_intermediates(self):
        return len(self.intermediates_names())
    def intermediates_names(self):
        names=["c0","c0act","c1","short","c1+short","act"]
        return names

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        short = self.shortcut(x)
        out+=short
        out = F.relu(out)

        return out

    def forward_intermediates(self,x):
        outputs=[]

        out=self.conv1(x)
        outputs.append(out)
        out = F.relu(self.bn1())
        outputs.append(out)

        out=self.conv2(out)
        outputs.append(out)
        out = F.relu(self.bn2())
        outputs.append(out)

        out=self.conv3(out)
        outputs.append(out)
        out = self.bn3(out)

        short = self.shortcut(x)
        outputs.append(short)
        out += short
        outputs.append(out)
        out = F.relu(out)
        outputs.append(out)
        return out,outputs

    def n_intermediates(self):
        return len(self.intermediates_names())
    def intermediates_names(self):
        names=["c0","c0act","c1","c1act","c2","short","c2+short","act"]
        return names

class ResNet(nn.Module):
    def __init__(self, block, num_blocks,input_shape,num_classes):
        super(ResNet, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64
        h,w,c=input_shape
        self.conv1 = nn.Conv2d(c, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)
        return out

    def forward_intermediates(self,x):
        outputs=[]
        x = self.conv1(x)
        outputs.append(x)
        x = F.relu(self.bn1(x))
        outputs.append(x)
        # print(x.shape)
        x,intermediates = self.layer_intermediates(self.layer1,x)
        outputs+=intermediates
        # print(x.shape)
        x,intermediates = self.layer_intermediates(self.layer2,x)
        outputs += intermediates
        # print(x.shape)
        x, intermediates = self.layer_intermediates(self.layer3,x)
        outputs += intermediates
        # print(x.shape)
        x, intermediates = self.layer_intermediates(self.layer4,x)
        outputs += intermediates
        # print(x.shape)
        x = F.avg_pool2d(x, 4)
        # print(x.shape)
        outputs.append(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)
        outputs.append(x)
        x = F.log_softmax(x, dim=1)
        outputs.append(x)
        return x,outputs
    def layer_intermediates(self,layer,x):
        outputs=[]
        for block in layer:
            x,intermediates=block.forward_intermediates(x)
            outputs+=intermediates
        # print(outputs)
        return x,outputs

    def n_intermediates(self):
        return len(self.intermediates_names())
    def intermediates_names(self):
        names=["c0","c0act"]
        for i,l in enumerate([self.layer1,self.layer2,self.layer3,self.layer4]):
            for j,block in enumerate(l):
                l_names=[f"l{i}_b{j}_{n}" for n in block.intermediates_names()]
                names.extend(l_names)
        names+=["avgp","fc0","fc0act"]
        return names

def ResNet18(input_shape,num_classes):
    return ResNet(BasicBlock, [2,2,2,2],input_shape,num_classes)

def ResNet34(input_shape,num_classes):
    return ResNet(BasicBlock, [3,4,6,3],input_shape,num_classes)

def ResNet50(input_shape,num_classes):
    return ResNet(Bottleneck, [3,4,6,3],input_shape,num_classes)

def ResNet101(input_shape,num_classes):
    return ResNet(Bottleneck, [3,4,23,3],input_shape,num_classes)

def ResNet152(input_shape,num_classes):
    return ResNet(Bottleneck, [3,8,36,3],input_shape,num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32),(40,40,3),10)
    print(y.size())
