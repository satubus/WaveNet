import torch


class CausalConv1d(torch.nn.Module):
    """
    A causal 1D convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, bias=False):
        super(CausalConv1d, self).__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1),
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
        return self.conv1d(x)[:, :, :-self.conv1d.padding[0]]


class ResidualConv1dGatedUnit(torch.nn.Module):
    def __init__(self, residual_channels, gate_channels, kernel_size,
                 skip_out_channels, cin_channels, gin_channels, padding, dilation, bias=True, causal=True):
        super(ResidualConv1dGatedUnit, self).__init__()
        # dilated convolution
        self.causal = causal
        self.conv = CausalConv1d(in_channels=residual_channels,
                                 out_channels=gate_channels,
                                 kernel_size=kernel_size, padding=padding,
                                 dilation=dilation, bias=bias)

        # conditioning
        if cin_channels > 0:
            self.conv1d1x1c = torch.nn.Conv1d(cin_channels, gate_channels, bias=False)
        else:
            self.conv1d1x1c = None

        if gin_channels > 0:
            self.conv1d1x1g = torch.nn.Linear(gin_channels, gate_channels, bias=False)
        else:
            self.conv1d1x1g = None

        # gate_out_channels =
        self.conv_res = None
        self.conv_skip = None

    def forward(self, x, h=None):
        """
        Forward.
        :param x: (Tensor) B x C x T
        :param h: (Tensor) B x C x T, Local conditioning features, expected to be already upsampled.
        :return:
        """

        residual = x
        # dilated conv
        x = self.conv(x)
        # split features
        a, b = x.split(x.size(1) // 2, dim=1)

        ### Conditioning ###
        # global conditioning
        if h is not None:
            if self.conv1d1x1g is not None:
                h = self.conv1d1x1g(h)
            else:
                h = self.conv1d1x1c(h)

            ha, hb = h.split(h.size(1) // 2, dim=1)
            a, b = a + ha, b + ha

        # What does local conditioning affect? The activation functions
        # When do we apply local conditioning and where?
        # What size does h have?
        # local conditioning
        # 1x1 conv
        # assume h is already up-sampled here
        h = self.conv1d1x1c(h)
        ha, hb = h.split(x.size(1) // 2, dim=1)

        # up-sampling for lower dim sample.

        # activation
        z = torch.tanh(x) * torch.sigmoid(x)
