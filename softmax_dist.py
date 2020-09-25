import torch


def softmax_distribution(xt, mu=255):
    """
    Softmax distribution, categorized into 256 mu with u-law analog to digital conversion method.
    @param mu: categories, default 256 like in paper.
    @param xt: input sample of wave.
    """

    return torch.sign(xt)*torch.log(1 + mu * torch.abs(xt))/torch.log(1 + mu)

if __name__ == '__main__':
    for i in range(256):
        print(softmax_distribution(torch.tensor(50),torch.tensor(i,dtype=float)))
