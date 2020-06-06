import torch
import torch.nn as nn

class L1_TVLoss_Charbonnier(nn.Module):

    def __init__(self):

        super(L1_TVLoss_Charbonnier, self).__init__()

        self.e = 0.000001 ** 2



    def forward(self, x):

        batch_size = x.size()[0]

        h_tv = torch.abs((x[:, :, 1:, :]-x[:, :, :-1, :]))

        h_tv = torch.mean(torch.sqrt(h_tv ** 2 + self.e))

        w_tv = torch.abs((x[:, :, :, 1:]-x[:, :, :, :-1]))

        w_tv = torch.mean(torch.sqrt(w_tv ** 2 + self.e))

        return h_tv + w_tv
