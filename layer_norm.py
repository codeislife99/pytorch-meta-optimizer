import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LayerNorm1D(nn.Module):

    def __init__(self, num_outputs, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.eps = eps
        # Create Modifiable Parameters
        self.weight = nn.Parameter(torch.ones(1, num_outputs)) 
        self.bias = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, inputs):
        # Taking the mean/std of columns.
        # keepdim = true will preserve the dimensionality of the input else input will be squeezed
        # expand_as will preserve the same size as input and repeat the mean/std along the columns(1st dimension)  
        input_mean = inputs.mean(1,keepdim=True).expand_as(inputs)
        input_std = inputs.std(1,keepdim=True).expand_as(inputs)
        x = (inputs - input_mean) / (input_std + self.eps) 
        # No of columns in x is equal to the num_outputs. 
        # This expand_as tries to expand the 1st row of self.weight/self.bias into the no of rows of x. 
        return x * self.weight.expand_as(x) + self.bias.expand_as(x)
