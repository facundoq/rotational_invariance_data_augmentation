from pytorch.model.old_conv import *
from pytorch.model.simple_conv import *
from pytorch.model.gconv import *
from pytorch.model.stn import *
from pytorch.model.all_conv import *
from pytorch.model.vgg_like import *
from pytorch.model.resnet import *


class FFNet(nn.Module):
    def __init__(self,input_shape,num_classes,h1=120,h2=84):
        super(FFNet, self).__init__()
        h,w,channels=input_shape

        self.linear_size = h * w * channels
        self.fc= nn.Sequential(
                nn.Linear(self.linear_size, 120),
             #   nn.BatchNorm1d(120),
                nn.ReLU(),
                nn.Linear(h1, h2),
              #  nn.BatchNorm1d(84),
                nn.ReLU(),
                nn.Linear(h2, num_classes)
                )

    def forward(self, x):
        x = x.view(-1, self.linear_size)
        x = self.fc(x)

        return x

