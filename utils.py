import torch


def mu_law_encoding(xt, mu=255):
    """
    Mu law analog to digital conversion.

    @param mu: categories, default 256 like in paper.
    @param xt: input sample of wave.
    """

    return torch.sign(xt) * torch.log(1 + mu * torch.abs(xt)) / torch.log(1 + mu)


def mu_law_deconding(xt, mu=255):
    """
    Mu-law decoding method.

    :param xt:
    :param mu:
    :return:
    """
    pass


def upsampling(in_channels, out_channels):
    """
    Getting an input feature and upsample it.
    TODO:
    Research how upsampling is done.
    Transposed convolutional neural network.
    Up_conv matrix is learned from data? Or is it hand engineered?
    Should I implement this as own function or integrate it into the dilated causal gate unit??
    No it should be upsampled before feeding as conditioning signal,
    so that means it could be a module outside the gated unit. What alternative placing does this code have?
    Integrate is as section of the architecture.
    :return:
    """
    torch.nn.ConvTranspose1d(in_channels, out_channels)


if __name__ == '__main__':
    for i in range(256):
        print(mu_law_encoding(torch.tensor(50), torch.tensor(i, dtype=float)))
