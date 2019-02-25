import torch.nn as nn
import torch.nn.functional as F


class OldConv(nn.Module):
    def __init__(self, input_shape, num_classes, filters=16,dense_filters=128,stacks=1,layers_per_stack=1):
        super(OldConv, self).__init__()
        self.name = "simple_conv"
        h, w, channels = input_shape
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(channels, filters, kernel_size=3, padding=1))
        self.conv_layers.append(nn.ReLU())
        # self.conv_layers.append(nn.BatchNorm2d(channels))

        self.stacks = stacks
        self.layers_per_stack = layers_per_stack

        for i in range(self.stacks):
            stack_filters = filters * (i + 1)
            next_stack_filters = filters * (i + 2)
            for j in range(self.layers_per_stack):
                name = "%d_%d_conv" % (i, j)

                self.conv_layers.append(nn.Conv2d(stack_filters, stack_filters, kernel_size=3, padding=1))
                self.conv_layers.append(nn.ReLU())
                # self.conv_layers.append(nn.BatchNorm2d(stack_filters))

            name = "%d_conv_stride" % i
            self.conv_layers.append(nn.Conv2d(stack_filters, next_stack_filters, kernel_size=3, stride=2, padding=1))
            self.conv_layers.append(nn.ReLU())
            # self.conv_layers.append(nn.BatchNorm2d(next_stack_filters))

        hf, wf = h // (2 ** self.stacks), w // (2 ** self.stacks)
        flattened_output_size = hf * wf * next_stack_filters

        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(nn.Linear(flattened_output_size, dense_filters))
        self.dense_layers.append(nn.ELU())
        # self.dense_layers.append(nn.BatchNorm1d(dense_filters))
        self.dense_layers.append(nn.Linear(dense_filters, num_classes))

    def forward(self, x):
        for l in self.conv_layers:
            x=l(x)
        x = x.view(x.size()[0], -1)
        for l in self.dense_layers:
            x=l(x)
        x=F.log_softmax(x, dim=1)
        return x
