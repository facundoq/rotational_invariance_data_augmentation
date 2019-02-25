import torch.nn as nn
import torch.nn.functional as F


from pytorch.model.util import SequentialWithIntermediates

class SimpleConv(nn.Module):


    def __init__(self,input_shape,num_classes,conv_filters=32,fc_filters=128):
        super(SimpleConv, self).__init__()
        self.name=self.__class__.__name__
        h,w,channels=input_shape

        # self.conv=nn.Sequential(
        #     nn.Conv2d(channels, conv_filters, 3,padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(conv_filters, conv_filters, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(conv_filters, conv_filters*2, 3, padding=1,stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(conv_filters*2, conv_filters*2, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(conv_filters*2, conv_filters*4, 3, padding=1,stride=2),
        #     nn.ReLU(),
        #     )

        self.conv = SequentialWithIntermediates(
            nn.Conv2d(channels, conv_filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_filters, conv_filters, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2,kernel_size=2),
            nn.Conv2d(conv_filters, conv_filters * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_filters * 2, conv_filters * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2,kernel_size=2),
            nn.Conv2d(conv_filters * 2, conv_filters * 4, 3, padding=1),
            nn.ReLU(),
        )

        self.linear_size = h * w * (conv_filters*4) // 4 // 4

        self.fc= SequentialWithIntermediates(
                nn.Linear(self.linear_size, fc_filters),
                # nn.BatchNorm1d(fc_filters),
                nn.ReLU(),
                nn.Linear(fc_filters, num_classes)
                )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.linear_size)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def forward_intermediates(self,x):
        x1, convs = self.conv.forward_intermediates(x)

        x2 = x1.view(-1, self.linear_size)
        x3,fcs = self.fc.forward_intermediates(x2)
        x4 = F.log_softmax(x3, dim=-1)
        return x4,convs+fcs+[x4]

    def n_intermediates(self):
        return len(self.intermediates_names())

    def intermediates_names(self):
        conv_layer_names = ["c1","c1act","c2", "c2act", "mp1",
                            "c3", "c3act", "c4", "c4act","mp2",
                            "c5", "c5act"]
        fc_layer_names = ["fc1", "fc1act", "fc2", "fc2act"]

        return conv_layer_names+fc_layer_names

    def layer_names(self):
        conv_layer_names = ["c1", "c2", "c3", "c4", "c5"]
        fc_layer_names = ["fc1", "fc2"]
        # bn_layer_names = ["bn"]
        layer_names = conv_layer_names + fc_layer_names #+ bn_layer_names
        return layer_names

    def layers(self):
        conv_layers = list(self.conv.children())
        conv_layers = subset(conv_layers, [0, 2, 5, 7, 10])
        fc_layers_all = list(self.fc.children())
        fc_layers = subset(fc_layers_all, [0, 2])
        # bn_layers = subset(fc_layers_all, [1])
        layers = conv_layers + fc_layers #+ bn_layers
        return layers

    def get_layer(self,layer_name):
        layers=self.layers()
        layer_names=self.layer_names()
        index=layer_names.index(layer_name)
        return layers[index]

def subset(l, indices):
    return [l[i] for i in indices]
