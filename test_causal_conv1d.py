import unittest
import torch
from causal_conv1d import CausalConv1d


class TestCausalConv1d(unittest.TestCase):
    """
    Unit Testing class for causal convolution
    """

    def test_one_layer(self):
        # setup causal conv
        causal_conv = CausalConv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            dilation=1
        )

        # Test case: one layer.
        weights = torch.Tensor([0, 2]).unsqueeze(0).unsqueeze(0)
        input = torch.Tensor([3, 4]).unsqueeze(0).unsqueeze(0)
        causal_conv.conv1d.weight.data = weights
        output = causal_conv(input)
        solution = torch.Tensor([6,8]).unsqueeze(0).unsqueeze(0)
        self.assertEqual(output.sum(), solution.sum())


if __name__ == '__main__':
    unittest.main()