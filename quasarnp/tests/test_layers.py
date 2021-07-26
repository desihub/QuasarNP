import math
import unittest

import numpy as np

from quasarnp.layers import (batch_normalization, conv1d, dense, flatten,
                             linear, relu, sigmoid)


class TestActivations(unittest.TestCase):
    def test_linear(self):
        # Test the liner activation for integers, positive and negative
        self.assertEqual(linear(0), 0)
        self.assertEqual(linear(1), 1)
        self.assertEqual(linear(-1), -1)

        # Test the linear activation for numpy arrays, both flat and 2d
        in_arr = np.arange(-10, 10)
        expected = np.arange(-10, 10)

        self.assertTrue(np.allclose(linear(in_arr), expected))

        in_arr = np.reshape(in_arr, (-1, 5))
        expected = np.reshape(expected, (-1, 5))
        self.assertTrue(np.allclose(linear(in_arr), expected))

    def test_relu(self):
        # Test the relu activation for integers, positive and negative
        self.assertEqual(relu(0), 0)
        self.assertEqual(relu(1), 1)
        self.assertEqual(relu(-1), 0)

        # Test the relu activation for numpy arrays, both flat and 2d
        in_arr = np.arange(-10, 10)
        expected = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1, 2, 3, 4, 5, 6, 7, 8, 9])

        self.assertTrue(np.allclose(relu(in_arr), expected))

        in_arr = np.reshape(in_arr, (-1, 5))
        expected = np.reshape(expected, (-1, 5))
        self.assertTrue(np.allclose(relu(in_arr), expected))

    def test_sigmoid(self):
        # Test the sigmoid activation for integers, positive and negative
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(1), 1 / (1 + 1 / math.e))
        self.assertEqual(sigmoid(-1), 1 / (1 + math.e))

        # Test the relu activation for numpy arrays, both flat and 2d
        in_arr = np.arange(-10, 10)
        expected = 1 / (1 + np.exp(-in_arr))

        self.assertTrue(np.allclose(sigmoid(in_arr), expected))

        in_arr = np.reshape(in_arr, (-1, 5))
        expected = np.reshape(expected, (-1, 5))
        self.assertTrue(np.allclose(sigmoid(in_arr), expected))


class TestLayers(unittest.TestCase):
    # For this test class we will assume that the activations are correct.
    # We test those separately up above anyway.
    def test_dense_weights(self):
        # This test has all weights set to 0, but nonzero/nonunity bias
        # This should mean the result is only the bias relu'd.
        in_weights = np.zeros((11, 5))
        in_bias = np.arange(-2, 3)
        in_x = np.arange(-5, 6)

        observed = dense(in_x, in_weights, in_bias, relu)
        expected = [0, 0, 0, 1, 2]
        self.assertTrue(np.allclose(observed, expected))

        # Setting the weights to 1 and ensuring the answer remains correct.
        # Since the input x array is symmetric the pre bias answer is still
        # zeros.
        in_weights = np.ones((11, 5))
        observed = dense(in_x, in_weights, in_bias, relu)
        expected = [0, 0, 0, 1, 2]
        self.assertTrue(np.allclose(observed, expected))

    def test_dense_bias(self):
        # This test has all biases set to 0, but nonzero/nonunity weights
        in_weights = [[1, 1, -5, -1, -1],
                      [1, 2, -4, -1, -2],
                      [1, 3, -3, -1, -3],
                      [1, 4, -2, -1, -4],
                      [1, 5, -1, -1, -5]]
        in_weights = np.asarray(in_weights)
        in_bias = np.zeros(5)
        in_x = np.arange(-2, 3)

        observed = dense(in_x, in_weights, in_bias, relu)
        expected = [0, 10, 10, 0, 0]
        self.assertTrue(np.allclose(observed, expected))

        # Testing if we set the bias to 1 that the answer remains correct
        # NOTE: The last weights product equals -10 so adding the
        # 1 does not change the answer of the relu.
        in_bias = np.ones(5)
        observed = dense(in_x, in_weights, in_bias, relu)
        expected = [1, 11, 11, 1, 0]
        self.assertTrue(np.allclose(observed, expected))

    # For testing flatten we need to test that the
    # array only flattens everything except the first dimension AND
    # that the flatten flattens in the correct order
    # We split the tests by dimensionality to help diagnose any problems.
    def test_flatten_2d(self):
        # Creates array that looks like this:
        # [1, 1, 1]
        # [2, 2, 2]
        # [3, 3, 3]
        column = np.asarray([1, 2, 3])
        in_x = np.asarray([column] * 3).T

        # "Flatten" which does nothing here since we flatten higher dimensions
        observed = flatten(in_x)
        expected = in_x

        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

    def test_flatten_3d(self):
        # Creates the following 3d array:
        # [[[0 1], [2 3]],[[4 5],[6 7]]]
        in_x = np.arange(0, 2 * 2 * 2).reshape((2, 2, 2))

        # Flatten and test
        observed = flatten(in_x)
        expected = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]])

        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

    def test_flatten_4d(self):
        # Creates the following 4d array:
        # [[[[0 1], [2 3]], [[4 5], [6 7]]],
        # [[[8 9], [10 11]], [[12 13], [14 15]]]]
        in_x = np.arange(0, 2 * 2 * 2 * 2).reshape((2, 2, 2, 2))

        # Flatten and test
        observed = flatten(in_x)
        expected = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
        expected = np.asarray(expected)

        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

    # This tests  batchnormalizing a scalar
    def test_batch_normalization_scalar(self):
        eps = 1e-7  # To avoid divide by zero errors
        x = 3
        mu = 2
        var = 2
        gamma = 1
        beta = 0

        # First test gamma = 1, beta = 0 so no scaling or offset
        observed = batch_normalization(x, mu, var, beta, gamma, eps)
        expected = 1 / np.sqrt(2)
        self.assertTrue(np.allclose(observed, expected))

        # Scale by double
        gamma = 2
        observed = batch_normalization(x, mu, var, beta, gamma, eps)
        expected = 2 / np.sqrt(2)
        self.assertTrue(np.allclose(observed, expected))

        # Offset by one and scale by double.
        beta = 1
        observed = batch_normalization(x, mu, var, beta, gamma, eps)
        expected = 2 / np.sqrt(2) + 1
        self.assertTrue(np.allclose(observed, expected))

    # We here use a 2d vector instead of a scalar to make sure the array logic
    # works out correctly
    def test_batch_normalization_vector(self):
        eps = 1e-7  # To avoid divide by zero errors
        x = np.asarray([3, 2, 1])
        mu = np.asarray([2, 2, 2])
        var = np.asarray([2, 2, 2])
        gamma = 1
        beta = 0

        # First test gamma = 1, beta = 0 so no scaling or offset
        observed = batch_normalization(x, mu, var, beta, gamma, eps)
        expected = [1 / np.sqrt(2), 0, -1 / np.sqrt(2)]
        self.assertTrue(np.allclose(observed, expected))

        # Scale by double
        gamma = 2
        observed = batch_normalization(x, mu, var, beta, gamma, eps)
        expected = [2 / np.sqrt(2), 0, -2 / np.sqrt(2)]
        self.assertTrue(np.allclose(observed, expected))

        # Offset by one and scale by double.
        beta = 1
        observed = batch_normalization(x, mu, var, beta, gamma, eps)
        expected = [2 / np.sqrt(2) + 1, 1, -2 / np.sqrt(2) + 1]
        self.assertTrue(np.allclose(observed, expected))

    def test_conv1d_base(self):
        # Shape (batch_size, in_width, in_channels)
        x = np.ones((5, 200, 5))

        # Shape (filter_width, in_channels, out_channels)
        w = np.ones((5, 5, 5))

        # We expect that with the 25 filters that the convolution will give 25
        # i.e. the sum of 25 ones
        observed = conv1d(x, w)
        expected = np.ones((5, 196, 5)) * 25
        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

        # Doubling the input to make extra sure it's good
        x = x * 2
        observed = conv1d(x, w)
        expected = np.ones((5, 196, 5)) * 50
        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

        # Testing a more dynamic x array.
        x = np.asarray([[1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5]]).reshape(3, -1, 1)
        w = np.ones((3, 1, 2))
        observed = conv1d(x, w)
        expected = [[[6, 6], [9, 9], [12, 12]],
                    [[6, 6], [9, 9], [12, 12]],
                    [[6, 6], [9, 9], [12, 12]]]
        expected = np.asarray(expected)
        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

    # Same test as above, but where we set the stride to 2 instead of 1.
    def test_conv1d_stride(self):
        # Shape (batch_size, in_width, in_channels)
        x = np.ones((5, 200, 5))

        # Shape (filter_width, in_channels, out_channels)
        w = np.ones((5, 5, 5))

        # We expect that with the 25 filters that the convolution will give 25
        # i.e. the sum of 25 ones
        observed = conv1d(x, w, stride=2)
        expected = np.ones((5, 98, 5)) * 25
        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

        # Doubling the input to make extra sure it's good
        x = x * 2
        observed = conv1d(x, w, stride=2)
        expected = np.ones((5, 98, 5)) * 50
        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

        # Testing a more dynamic x array.
        x = np.asarray([[1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5]]).reshape(3, -1, 1)
        w = np.ones((3, 1, 2))
        observed = conv1d(x, w, stride=2)
        expected = [[[6, 6], [12, 12]],
                    [[6, 6], [12, 12]],
                    [[6, 6], [12, 12]]]
        expected = np.asarray(expected)
        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

    # Same test as base, but we add a bias vector of ones.
    def test_conv1d_bias(self):
        # Shape (batch_size, in_width, in_channels)
        x = np.ones((5, 200, 5))

        # Shape (filter_width, in_channels, out_channels)
        w = np.ones((5, 5, 5))

        # Shape (out_channels)
        b = np.ones(5)

        # We expect that with the 25 filters that the convolution will give 25
        # i.e. the sum of 25 ones, plus then the additional 1 from the bias
        observed = conv1d(x, w, b=b)
        expected = np.ones((5, 196, 5)) * 26
        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

        # Doubling the input to make extra sure it's good
        x = x * 2
        observed = conv1d(x, w, b=b)
        expected = np.ones((5, 196, 5)) * 51
        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))

        # Testing a more dynamic x array.
        x = np.asarray([[1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5]]).reshape(3, -1, 1)
        w = np.ones((3, 1, 2))
        b = np.ones(2)
        observed = conv1d(x, w, b=b)
        expected = [[[7, 7], [10, 10], [13, 13]],
                    [[7, 7], [10, 10], [13, 13]],
                    [[7, 7], [10, 10], [13, 13]]]
        expected = np.asarray(expected)
        self.assertEqual(observed.shape, expected.shape)
        self.assertTrue(np.allclose(observed, expected))


if __name__ == '__main__':
    unittest.main()
