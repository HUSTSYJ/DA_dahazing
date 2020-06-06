import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms 
from .grad_conv import grad_conv_hor, grad_conv_vet
from torch.nn.functional import l1_loss

# img must be variable with grad and of dim N*C*W*H
def TVLossL1(img):
    hor = grad_conv_hor()(img)
    vet = grad_conv_vet()(img)
    target = torch.autograd.Variable(torch.FloatTensor(img.shape).zero_().cuda())
    loss_hor = l1_loss(hor, target, size_average=False)
    loss_vet = l1_loss(vet, target, size_average=False)
    loss = loss_hor+loss_vet
    return loss
    

if __name__ == "__main__":
    img = Image.open('1.jpg')
    img = transforms.ToTensor()(img)[None, :, :, :]
    img = torch.autograd.Variable(img, requires_grad=True)
    
    loss = TVLossL1(img)
