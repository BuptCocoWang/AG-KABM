import torch
import torch.nn as nn
import torch.nn.functional as F

class ELA(nn.Module):
    def __init__(self, in_channels):
        super(ELA, self).__init__()

        Kernel_size = 5
        groups = in_channels
        num_groups = 32
        pad = Kernel_size // 2

        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input):
        b, c, h, w = input.size()
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)

        x_h = self.con1(x_h)  # [b, c, h]
        x_w = self.con1(x_w)  # [b, c, w]

        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)  # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)  # [b, c, 1, w]
        return x_h * x_w * input