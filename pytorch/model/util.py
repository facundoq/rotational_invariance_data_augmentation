import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SequentialWithIntermediates(nn.Sequential):
    def __init__(self,*args):
        super(SequentialWithIntermediates, self).__init__(*args)

    def forward_intermediates(self,input_tensor):
        outputs=[]
        for module in self._modules.values():
            input_tensor= module(input_tensor)
            outputs.append(input_tensor)
        return input_tensor,outputs