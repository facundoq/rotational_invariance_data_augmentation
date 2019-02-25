import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from pytorch.model.simple_conv import SimpleConv
from .util import Flatten
from .all_conv import AllConvolutional,AllConv




class STN(nn.Module):
    def __init__(self, input_shape, num_classes,only_rotation, net=None,locnet=None):
        super(STN, self).__init__()

        self.plot = False
        if net is None:
            net=SimpleConv(input_shape, num_classes)

        h, w, channels = input_shape
        conv_filters = 16
        size_change=lambda x: x//(2*2)
        flat_size = size_change(h) * size_change(w) * conv_filters
        linear_size=32

        if locnet is None:
            locnet = nn.Sequential(
                nn.Conv2d(channels, conv_filters, kernel_size=7, padding=3),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(),
                nn.Conv2d(conv_filters, conv_filters, kernel_size=5, padding=2),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(),
                Flatten(),
                nn.Linear(flat_size, linear_size),
            )
        if only_rotation:
            self.stl = LearnableRotationTransformation(locnet,locnet_output_size_flattened=linear_size,add_top=True)
        else:
            self.stl = LearnableAffineTransformation(locnet,locnet_output_size_flattened=linear_size,add_top=True)
        self.net=net
        self.count=0

    def plot_batch(self, before,after):
        n,c,h,w=before.shape
        columns=8
        rows=n // columns + (n % columns>0)

        combination=np.empty((n,c,h,w*2))
        combination[:,:,:,:h]=before
        combination[:, :, :, h:] = after
        combination=combination.transpose(0,2,3,1)

        f, plots = plt.subplots(rows,columns,figsize=(10,4),dpi=100)

        for i in range(rows*columns):
            row, col = i // columns, i % columns
            if i<n:
                if c == 1:
                    plots[row,col].imshow(combination[i, :, :, 0])
                else:
                    plots[row,col].imshow(combination[i,:,:,:])
            plots[row, col].axis("off")
        plt.show()

    def forward(self, x):
        if self.count==0 and self.plot:
            x_before = x[:, :, :, :].clone().cpu().detach().numpy()
            x=self.stl(x)
            x_after = x[:, :, :, :].clone().cpu().detach().numpy()
            self.plot_batch(x_before,x_after)
        else:
            x = self.stl(x)
        self.count+=1
        x=self.net(x)
        return x


class SimpleSTN(STN):
    def __init__(self, input_shape, num_classes, conv_filters,fc_filters,only_rotation=True):
        net = SimpleConv(input_shape, num_classes, conv_filters=conv_filters, fc_filters=fc_filters)
        super(SimpleSTN, self).__init__(input_shape, num_classes,only_rotation,net=net)
        self.name="SimpleConvSTN"


class AllConvSTN(STN):
    def __init__(self, input_shape, num_classes, filters,only_rotation=True):
        net = AllConv(input_shape, num_classes,filters=filters)
        super(AllConvSTN, self).__init__(input_shape, num_classes,only_rotation,net=net)
        self.name="AllConvolutionalSTN"



class LearnableAffineTransformation(nn.Module):

    def __init__(self,locnet,add_top=True,locnet_output_size_flattened=None):
        super(LearnableAffineTransformation, self).__init__()
        self.locnet=locnet
        self.add_top=add_top
        self.use_cuda = True

        if self.add_top:
            assert locnet_output_size_flattened
            self.fc_loc = nn.Linear(locnet_output_size_flattened, 6)

            # Initialize the weights/bias with identity transformation
            self.fc_loc.weight.data.zero_()
            t = torch.tensor([0], dtype=torch.float)
            self.fc_loc.bias.data.copy_(t)

            # Initialize the weights/bias with identity transformation
            self.fc_loc.weight.data.zero_()
            self.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def cuda(self):
        self.use_cuda = True
        return super(LearnableAffineTransformation, self).cuda()


    def forward(self,x):
        theta=self.locnet(x)
        if self.add_top:
            # flatten output
            theta = theta.view(x.size()[0], -1)
            theta = self.fc_loc(theta)
            theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x_t = F.grid_sample(x, grid)
        return x_t



# A layer for the STN with rotation correction only
class LearnableRotationTransformation(nn.Module):
    def __init__(self,locnet,add_top=True,locnet_output_size_flattened=None):
        super(LearnableRotationTransformation, self).__init__()
        self.locnet=locnet
        self.add_top=add_top
        self.use_cuda=True
        if self.add_top:
            assert locnet_output_size_flattened
            self.fc_loc = nn.Linear(locnet_output_size_flattened, 2)

            # Initialize the weights/bias with identity transformation
            self.fc_loc.weight.data.zero_()
            t=torch.tensor([3,0], dtype=torch.float)
            self.fc_loc.bias.data.copy_(t)

    # generate rotation matrix based on the angle for each batch sample

    def cuda(self):
        self.use_cuda=True
        return super(LearnableRotationTransformation,self).cuda()

    def trig_rotation_matrix(self,c,s):
        batch_size = c.shape[0]
        rotation_matrix = Variable(torch.zeros(batch_size, 2, 3), requires_grad=True)
        if self.use_cuda:
            rotation_matrix = rotation_matrix.cuda()
        rotation_matrix[:, 0, 0] = c
        rotation_matrix[:, 1, 1] = c
        rotation_matrix[:, 0, 1] = -s
        rotation_matrix[:, 1, 0] = s

        return rotation_matrix

    def cos_sin(self,x):
        theta = self.locnet(x)
        if self.add_top:
            theta = self.fc_loc(theta)
            theta = torch.tanh(theta)  # .squeeze()
            normalizing_factor = (theta ** 2).norm(2, 1, keepdim=True) + 1e-8
            theta = theta / normalizing_factor
        return theta

    def forward(self,x):
        theta = self.cos_sin(x)
        rotation_matrix=self.trig_rotation_matrix(theta[:,0],theta[:,1])
        # print(rotation_matrix)
        grid = F.affine_grid(rotation_matrix, x.size())
        x_t = F.grid_sample(x, grid)
        return x_t





class OriginalSTN(nn.Module):

    def __init__(self, input_shape, num_classes):
        super(OriginalSTN, self).__init__()
        self.name = "OriginalSTN"

        h, w, channels = input_shape

        self.conv1 = nn.Conv2d(channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stl(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stl(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

