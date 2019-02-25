# def convbnrelu(input,output):
#         return nn.Sequential(append(nn.Conv2d(input, output, kernel_size=3, padding=1))
#             ,self.conv_layers.append(nn.ELU())
#             ,self.conv_layers.append(nn.BatchNorm2d(output))
#         )

import torch.nn as nn
import torch.nn.functional as F
from pytorch.model.util import Flatten
from pytorch.model.util import SequentialWithIntermediates

class ConvBNRelu(nn.Module):

    def __init__(self,input,output):
        super(ConvBNRelu, self).__init__()
        self.name = "ConvBNRelu"
        self.layers=SequentialWithIntermediates(
            nn.Conv2d(input, output, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(output),
            )

    def forward(self,x):
        return self.layers.forward(x)

    def forward_intermediates(self,x):
        return self.layers.forward_intermediates(x)



class VGGLike(nn.Module):
    def __init__(self, input_shape, num_classes,conv_filters,fc):
        super(VGGLike, self).__init__()
        self.name = self.__class__.__name__
        h, w, channels = input_shape
        filters=conv_filters
        filters2=2*filters
        filters3=4*filters
        filters4=8*filters
        self.conv_layers = nn.Sequential(
            ConvBNRelu(channels, filters),
            ConvBNRelu(filters, filters),
            nn.MaxPool2d(2,2),
            ConvBNRelu(filters, filters2),
            ConvBNRelu(filters2, filters2),
            nn.MaxPool2d(2, 2),
            ConvBNRelu(filters2, filters3),
            ConvBNRelu(filters3, filters3),
            nn.MaxPool2d(2, 2),
            ConvBNRelu(filters3, filters4),
            ConvBNRelu(filters4, filters4),
            nn.MaxPool2d(2, 2),
            Flatten(),
        )
        max_pools=4
        hf, wf = h // (2 ** max_pools), w // (2 ** max_pools)
        flattened_output_size = hf * wf * filters4

        self.dense_layers = SequentialWithIntermediates(

            nn.Linear(flattened_output_size, fc),
            nn.BatchNorm1d(fc),
            nn.ReLU(),
            nn.Linear(fc,num_classes)

        )

    def forward(self, x):
        x=self.conv_layers.forward(x)
        x=self.dense_layers.forward(x)
        x=F.log_softmax(x, dim=1)
        return x
    def forward_intermediates(self,x):
        outputs = []
        for i in range(4):
            x,intermediates = self.conv_layers[i*3].forward_intermediates(x)
            outputs.append(intermediates[0])
            outputs.append(intermediates[1])
            x,intermediates = self.conv_layers[i*3+1].forward_intermediates(x)
            outputs.append(intermediates[0])
            outputs.append(intermediates[1])
            x = self.conv_layers[i*3+ 2].forward(x)
            outputs.append(x)
        x=self.conv_layers[-1].forward(x)# flatten
        x,intermediates=self.dense_layers.forward_intermediates(x)
        outputs.append(intermediates[0])
        outputs.append(intermediates[2])
        outputs.append(intermediates[3])
        x = F.log_softmax(x, dim=1)
        outputs.append(x)
        return x, outputs


    def n_intermediates(self):
        return len(self.intermediates_names())
    def intermediates_names(self):
        names=[]
        for i in range(4):
            names.append(f"c{i}_0")
            names.append(f"c{i}_0act")
            names.append(f"c{i}_1")
            names.append(f"c{i}_1act")
            names.append(f"mp{i}")
        names+=["fc1","fc1act","fc2","fc2act"]
        return names




