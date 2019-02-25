import torch.nn as nn
import torch.nn.functional as F
from pytorch.model.util import SequentialWithIntermediates

class AllConvolutional(nn.Module):
    def __init__(self, input_shape, num_classes=10,filters=96,dropout=False):
        super(AllConvolutional, self).__init__()
        self.name = self.__class__.__name__
        h,w,c=input_shape
        filters2=filters*2
        self.dropout=dropout
        self.conv1 = nn.Conv2d(c, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.conv3 = nn.Conv2d(filters, filters, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(filters, filters2, 3, padding=1)
        self.conv5 = nn.Conv2d(filters2, filters2, 3, padding=1)
        self.conv6 = nn.Conv2d(filters2, filters2, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(filters2, filters2, 3, padding=1)
        self.conv8 = nn.Conv2d(filters2, filters2, 1)
        self.class_conv = nn.Conv2d(filters2, num_classes, 1)

        self.layers= [self.conv1, self.conv2, self.conv3, self.conv4,
                         self.conv5, self.conv6, self.conv7, self.conv8,
                         self.class_conv]
    def forward(self, x):
        out,intermediates=self.forward_intermediates(x)
        return out

    def forward_intermediates(self, x):

        intermediates=[]
        dropout_probabilities=[.2,.5,.5]
        layer=0
        for i in range(3):
            if self.dropout:
                x = F.dropout(x, dropout_probabilities[i])
            for j in range(3):
                x=self.layers[layer](x)
                intermediates.append(x)
                x=F.relu(x)
                intermediates.append(x)
                layer+=1
        # x = F.dropout(x, .2)
        # conv1_out = F.relu(self.conv1(x))
        # conv2_out = F.relu(self.conv2(conv1_out))
        # conv3_out = F.relu(self.conv3(conv2_out))
        # if self.dropout:
        #     conv3_out= F.dropout(conv3_out, .5)
        # conv4_out = F.relu(self.conv4(conv3_out))
        # conv5_out = F.relu(self.conv5(conv4_out))
        # conv6_out = F.relu(self.conv6(conv5_out))
        # if self.dropout:
        #     conv6_out= F.dropout(conv6_out, .5)
        # conv7_out = F.relu(self.conv7(conv6_out))
        # conv8_out = F.relu(self.conv8(conv7_out))
        # class_out = F.relu(self.class_conv(conv8_out))

        pool_out = x.reshape(x.size(0), x.size(1), -1).mean(-1)
        # pool_out = F.adaptive_avg_pool2d(class_out, 1)
        # pool_out.squeeze_(-1)
        # pool_out.squeeze_(-1)

        log_probabilities=F.log_softmax(pool_out,dim=1)
        intermediates.append(pool_out)
        intermediates.append(log_probabilities)
        return log_probabilities,intermediates


    def layer_names(self):
        return [f"c{i}" for i in range(8)]+["class_conv"]


    # def layers(self):
    #     return [self.conv1,self.conv2,self.conv3,self.conv4,self.conv5,self.conv6,self.conv7,self.conv8,self.class_conv]

    def get_layer(self,layer_name):

        layer_names=self.layer_names()
        index=layer_names.index(layer_name)
        return self.layers[index]

    def n_intermediates(self):
        return len(self.intermediates_names())

    def intermediates_names(self):
        names=[ [x,x+"act"] for x in self.layer_names()]
        names = sum(names,[])
        names+=["avgpool","logsoftmax"]

        return names


class ConvBNAct(nn.Module):
    def __init__(self,in_filters,out_filters,stride=1,kernel_size=3):
        super(ConvBNAct, self).__init__()
        if kernel_size==0:
            padding=0
        else:
            padding=1
        self.model=nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=padding,stride=stride),
            nn.BatchNorm2d(out_filters),
            nn.ReLU()
        )

    def forward(self,x):
        return self.model.forward(x)

class AllConv(nn.Module):
    def __init__(self, input_shape, num_classes, filters=96):
        super(AllConv, self).__init__()
        self.name = self.__class__.__name__
        filters2 = filters * 2
        h, w, channels = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            ConvBNAct(filters, filters),
            ConvBNAct(filters, filters, stride=2),
            ConvBNAct(filters, filters2, ),
            ConvBNAct(filters2, filters2),
            ConvBNAct(filters2, filters2, stride=2),
            ConvBNAct(filters2, filters2),
            ConvBNAct(filters2, filters2, kernel_size=1),

        )

        final_channels = filters2

        self.class_conv = nn.Conv2d(final_channels, num_classes, 1)

    def forward(self, x):
        # # print(x.shape)
        # x = F.relu(self.conv1(x))
        # # print(x.shape)
        # x = F.relu(self.conv2(x))
        # # print(x.shape)
        # x = F.relu(self.conv3(x))
        # # print(x.shape)
        # x = F.relu(self.conv4(x))
        # # print(x.shape)
        # # print(x.shape)
        # x = F.relu(self.conv5(x))
        x = self.conv(x)

        class_out = F.relu(self.class_conv(x))
        pool_out = class_out.reshape(class_out.size(0), class_out.size(1), -1).mean(-1)
        # pool_out = F.adaptive_avg_pool2d(class_out, 1)
        # pool_out.squeeze_(-1)
        # pool_out.squeeze_(-1)

        log_probabilities = F.log_softmax(pool_out, dim=1)
        return log_probabilities

    def layer_names(self):
        return [f"conv{i}" for i in range(8)] + ["class_conv"]

    def layers(self):
        convs=list(self.conv.children())
        return [convs[0]]+[list(convs[i].model.children())[0] for i in range(3,10)]


        # return x
