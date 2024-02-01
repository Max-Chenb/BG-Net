import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    def __init__(self, d_model = 256, max_shape=(256, 256), temp_bug_fix=True):

        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :x.size(2), :x.size(3)]

if __name__ == '__main__':
    device = torch.device('cuda')
    net = PositionEncodingSine()
    net.to(device)
    x1 = torch.rand((2, 6, 128, 128)).cuda()
    x = net(x1)
    print(x.shape)

