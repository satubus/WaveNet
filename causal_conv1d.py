import torch


class CausalConv1d(torch.nn.Module):
    """
    A causal 1D convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size,  dilation, bias=False):
        super(CausalConv1d, self).__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size-1),
            dilation=dilation,
            bias=bias
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        :param x: Signal, 1D array/sequence
        :return:
        """
        # remove trailing padding
        return self.conv1d(x)[:,:,:-self.conv1d.padding[0]]

