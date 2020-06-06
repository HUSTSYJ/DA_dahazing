import torch
import numpy as np
from PIL import Image

import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.autograd import Variable
from torchvision import transforms
import pdb





def DCLoss(img, patch_size):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
    dc = maxpool(0-img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()) 
     
    loss = L1Loss(size_average=True)(-dc, target)
    return loss

def BCLoss(img, patch_size):
    """
    calculating bright channel of image, the image shape is of N*C*W*H
    """
    patch_size = 35
    dc = maxpool(img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()+1) 
    loss = L1Loss(size_average=False)(dc, target)
    return loss
    
if __name__=="__main__":
    img = Image.open('clear_img.jpg')
    totensor = transforms.ToTensor()
    
    img = totensor(img)
    
    img = Variable(img[None, :, :, :].cuda(), requires_grad=True)    
    loss = DCLoss(img, 35)
    
    # loss.backward()



    



