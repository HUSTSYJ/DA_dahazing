import torch
import torch.nn as nn
from torch.nn import Conv2d
import numpy as np

# horizontal gradient, the input_channel is default to 3
def grad_conv_hor():
	grad = Conv2d(3, 3, (1, 3), stride=1, padding=(0, 1))
	
	weight = np.zeros((3, 3, 1, 3))
	for i in range(3):
		weight[i, i, :, :] = np.array([[-1, 1, 0]])
	weight = torch.FloatTensor(weight).cuda()
	weight = nn.Parameter(weight, requires_grad=False)
	bias = np.array([0, 0, 0])
	bias = torch.FloatTensor(bias).cuda()
	bias = nn.Parameter(bias, requires_grad=False)
	grad.weight = weight
	grad.bias = bias
	return  grad

# vertical gradient, the input_channel is default to 3
def grad_conv_vet():
	grad = Conv2d(3, 3, (3, 1), stride=1, padding=(1, 0))
	weight = np.zeros((3, 3, 3, 1))
	for i in range(3):
		weight[i, i, :, :] = np.array([[-1, 1, 0]]).T
	weight = torch.FloatTensor(weight).cuda()
	weight = nn.Parameter(weight, requires_grad=False)
	bias = np.array([0, 0, 0])
	bias = torch.FloatTensor(bias).cuda()
	bias = nn.Parameter(bias, requires_grad=False)
	grad.weight = weight
	grad.bias = bias
	return  grad



