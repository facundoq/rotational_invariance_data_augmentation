from groupy.gconv.pytorch_gconv.splitgconv2d import P4MConvZ2, P4MConvP4M,P4ConvP4,P4ConvZ2
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
import torch.nn as nn
import torch.nn.functional as F

def plane_group_rotation_max_pooling(x):
    # xs = x.size()
    v, a=x.max(dim=2)
    return v

class SimpleGConv(nn.Module):
    # Pytorch:
    # https://github.com/adambielski/GrouPy
    # https://github.com/tscohen/gconv_experiments
    # Tensorflow
    # https://github.com/tscohen/GrouPy
    # https://github.com/tscohen/gconv_experiments
    def __init__(self,input_shape,num_classes,filters=16,fc_filters=32,pool_rotations=True):
        super(SimpleGConv, self).__init__()
        self.name="SimpleGConv"
        h,w,channels=input_shape
        self.pool_rotations=pool_rotations

        self.conv1 = P4ConvZ2(channels, filters, kernel_size=3,padding=1)
        self.conv2 = P4ConvP4(filters, filters, kernel_size=3,padding=1)
        self.conv3 = P4ConvP4(filters, filters, kernel_size=3,padding=1)
        self.conv4 = P4ConvP4(filters, filters*2, kernel_size=3,padding=1)
        self.conv5 = P4ConvP4(filters*2, filters * 4, kernel_size=3, padding=1)
        linear_size=(h // 4) * (w // 4) * filters * 4
        if not self.pool_rotations:
            linear_size*=4
        self.fc1 = nn.Linear(linear_size, fc_filters)
        self.bn1 = nn.BatchNorm1d(fc_filters)
        self.fc2 = nn.Linear(fc_filters, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = plane_group_spatial_max_pooling(x, 2, 2)
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv4(x))
        # print(x.shape)
        x = plane_group_spatial_max_pooling(x, 2, 2)
        # print(x.shape)
        x = F.relu(self.conv5(x))
        if self.pool_rotations:
            x = plane_group_rotation_max_pooling(x)
        # print(x.shape)
        x = x.view(x.size()[0], -1)
        # print(x.shape)
        x = F.relu(self.bn1(self.fc1(x)))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        # return x


class GroupAllConvolutional(nn.Module):
    def __init__(self, input_shape, num_classes=10,filters=96,pool_rotations=True):
        super(GroupAllConvolutional, self).__init__()
        h,w,c=input_shape
        self.pool_rotations=pool_rotations
        self.name="AllGConvolutional"
        filters2=filters*2
        self.conv1 = P4ConvZ2(c, filters, kernel_size=3, padding=1)
        self.conv2 = P4ConvP4(filters,filters, kernel_size=3, padding=1)
        self.conv2 = P4ConvP4(filters, filters, kernel_size=3, padding=1,stride=2)
        self.conv3 = P4ConvP4(filters, filters2, kernel_size=3, padding=1)
        self.conv4 = P4ConvP4(filters2, filters2, kernel_size=3, padding=1)

        self.conv5 = P4ConvP4(filters2, filters2, kernel_size=3, padding=1)
        self.conv6 = P4ConvP4(filters2, filters2, kernel_size=3, padding=1, stride=2)
        self.conv7 = P4ConvP4(filters2, filters2, kernel_size=3, padding=1)
        self.conv8 = P4ConvP4(filters2, filters2, kernel_size=1)

        if  self.pool_rotations:
            final_filters = filters2
        else:
            final_filters = filters2 * 4

        self.class_conv = nn.Conv2d(final_filters, num_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        # conv3_out = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        # conv6_out = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out))
        conv8_out = F.relu(self.conv8(conv7_out))
        if self.pool_rotations:
            conv8_out = plane_group_rotation_max_pooling(conv8_out)
            # print(x.shape)
        else:
            x=conv8_out
            conv8_out = x.view(x.size()[0], -1, x.size()[3], x.size()[4])

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = class_out.reshape(class_out.size(0), class_out.size(1), -1).mean(-1)

        return F.log_softmax(pool_out, dim=1)



class ConvBNAct(nn.Module):
    def __init__(self,in_filters,out_filters,stride=1,kernel_size=3):
        super(ConvBNAct, self).__init__()
        if kernel_size==0:
            padding=0
        else:
            padding=1
        self.model=nn.Sequential(
            P4ConvP4(in_filters, out_filters, kernel_size=kernel_size, padding=padding,stride=stride),
            #nn.BatchNorm2d(out_filters),
            nn.ReLU()
        )

    def forward(self,x):
        return self.model.forward(x)


class AllGConv(nn.Module):
    # Pytorch:
    # https://github.com/adambielski/GrouPy
    # https://github.com/tscohen/gconv_experiments
    # Tensorflow
    # https://github.com/tscohen/GrouPy
    # https://github.com/tscohen/gconv_experiments
    def __init__(self,input_shape,num_classes,filters=96,pool_rotations=True):
        super(AllGConv, self).__init__()
        self.name="AllGConvolutional"
        filters2=filters*2
        h,w,channels=input_shape
        self.pool_rotations=pool_rotations

        # self.conv1 = nn.Sequential(
        #         P4MConvZ2(channels, filters, kernel_size=3,padding=1),
        #         nn.BatchNorm2d(filters),
        #         nn.ReLU()
        #         )
        # self.conv2 = ConvBNAct(filters,filters)
        # self.conv3 = ConvBNAct(filters, filters,stride=2)
        # self.conv4 = ConvBNAct(filters, filters2, stride=2)
        # self.conv5 = ConvBNAct(filters2, filters2)
        # self.conv6 = ConvBNAct(filters2, filters2,  stride=2)
        # self.conv7 = ConvBNAct(filters2, filters2)
        # self.conv8 = ConvBNAct(filters2, filters2, kernel_size=1)
        self.conv = nn.Sequential(
             P4ConvZ2(channels, filters, kernel_size=3, padding=1),
             #nn.BatchNorm2d(filters),
             nn.ReLU(),
             ConvBNAct(filters, filters),
             ConvBNAct(filters, filters, stride=2),
             ConvBNAct(filters, filters2, ),
             ConvBNAct(filters2, filters2),
             ConvBNAct(filters2, filters2, stride=2),
             ConvBNAct(filters2, filters2),
             ConvBNAct(filters2, filters2, kernel_size=1),

        )
        final_channels=filters2
        if not self.pool_rotations:
            final_channels*=4

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
        if self.pool_rotations:
            x = plane_group_rotation_max_pooling(x)
            # print(x.shape)
        else:
            x = x.view(x.size()[0], -1, x.size()[3], x.size()[4])
        # print(x.shape)

        class_out = F.relu(self.class_conv(x))
        pool_out = class_out.reshape(class_out.size(0), class_out.size(1), -1).mean(-1)
        # pool_out = F.adaptive_avg_pool2d(class_out, 1)
        # pool_out.squeeze_(-1)
        # pool_out.squeeze_(-1)

        log_probabilities = F.log_softmax(pool_out, dim=1)
        return log_probabilities


        # return x


# class GroupAllConvolutional(nn.Module):
#     def __init__(self, input_shape, num_classes=10):
#         super(GroupAllConvolutional, self).__init__()
#         h,w,c=input_shape
#         self.conv1 = P4MConvZ2(c, 96, kernel_size=3, padding=1)
#         self.conv2 = P4MConvP4M(96,96, kernel_size=3, padding=1)
#         self.conv2 = P4MConvP4M(96, 96, kernel_size=3, padding=1,stride=2)
#         self.conv3 = P4MConvP4M(96, 192, kernel_size=3, padding=1)
#         self.conv4 = P4MConvP4M(192, 192, kernel_size=3, padding=1)
#
#         self.conv5 = P4MConvP4M(192, 192, kernel_size=3, padding=1)
#         self.conv6 = P4MConvP4M(192, 192, kernel_size=3, padding=1, stride=2)
#         self.conv7 = P4MConvP4M(192, 192, kernel_size=3, padding=1)
#         self.conv8 = P4MConvP4M(192, 192, kernel_size=1)
#         self.class_conv = nn.Conv2d(192*8, num_classes, 1)
#
#
#     def forward(self, x):
#         x_drop = F.dropout(x, .2)
#         conv1_out = F.relu(self.conv1(x_drop))
#         conv2_out = F.relu(self.conv2(conv1_out))
#         conv3_out = F.relu(self.conv3(conv2_out))
#         conv3_out_drop = F.dropout(conv3_out, .5)
#         conv4_out = F.relu(self.conv4(conv3_out_drop))
#         conv5_out = F.relu(self.conv5(conv4_out))
#         conv6_out = F.relu(self.conv6(conv5_out))
#         conv6_out_drop = F.dropout(conv6_out, .5)
#         conv7_out = F.relu(self.conv7(conv6_out_drop))
#         conv8_out = F.relu(self.conv8(conv7_out))
#
#         class_out = F.relu(self.class_conv(conv8_out))
#         pool_out = F.adaptive_avg_pool2d(class_out, 1)
#         pool_out.squeeze_(-1)
#         pool_out.squeeze_(-1)
#
#         return F.log_softmax(pool_out, dim=1)
