import torch
from causal_conv1d import CausalConv1d


if __name__ == '__main__':
    # Wie implementiert man so einen signal split in pytorch?
    # nvm.
    res1 = x
    x1 = torch.tanh(torch.conv1d(x,W)) * torch.sig(torch.conv1d(x,W))
    x2 = torch.conv1d(x1)
    x3 = x2 + x

    # incremental forward?