"""A module defining numpy implementations of the four layers used in QuasarNet.

In addition to the four layers defined here (dense, batch_normalization, conv1d
and flatten), this module defines three activation functions necessary to
replicate the behaviour of QuasarNet. These three activation functions are the
relu, linear and sigmoid activation functions.
"""
import numpy as np

# Useful
# https://www.tensorflow.org/guide/tensor

# Activations
relu = lambda x: np.maximum(0, x)
linear = lambda x: x
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Required layers
# Make sure that TF does x@w and not w@x
# ^ Confirmed via weights saved in hdf5 file
def dense(x, w, b, phi):
    """Computes a dense layer operation on the input data.

    This function implements a single dense layer for a neural network. The input
    data is dotted with the weights vector. The bias is added to the output.
    Finally, the activation function is run over the data and the result is returned.

    Parameters
    ----------
    x : numpy.ndarray
        Input data.
    w : numpy.ndarray
        Weights vector.
    b : numpy.ndarray
        Bias vector.
    phi : function
        Activation function to apply to the output.

    Returns
    -------
    numpy.ndarray
        Output of the dense layer.
    """
    return phi(x @ w + b)

# This uses the "moving mean" which is compiled from the training set
# rather than just taking the mean of the input data like I would assume.
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
# This has same behavior as:
# https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
def batch_normalization(x, mean, var, beta, gamma, epsilon):
    """Computes the batch normalized version of the input.

    This function implements a batch normalization layer. Batch normalization
    renormalizes the input to the layer to a more parsable data range.

    Parameters
    ----------
    x : numpy.ndarray
        Input data.
    mean : numpy.ndarray
        Moving mean of the dataset, computed during training.
    var : numpy.ndarray
        Moving variance of the dataset, computer during training.
    beta : array_like
        Offset value added to the normalized output.
    gamma : array_like
        Scale value to rescale the normalized output.
    epsilon : float
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Output of batch normalization.

    Notes
    -----
    The operation implemented in this function is:

    .. math:: \\frac{\gamma (x - \mu)}{\sigma} + \\beta

    where :math:`\mu` is the moving mean of the dataset and :math:`\sigma` is
    the moving variance of the dataset, both of which are computed during
    training.

    More details and documentation on the TensorFlow batch_normalization
    function that this function mimics can be found at
    https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization.
    """
    return ((x - mean) / np.sqrt(var + epsilon)) * gamma + beta

# This assumes x and w are in the exact form you would pass
# them to tf.nn.conv1d for example
# This has same behavior as:
# https://www.tensorflow.org/api_docs/python/tf/nn/conv1d
def conv1d(x, w, stride=1, b=None):
    """Computes a 1-d convolution given 3-d input and weight arrays.

    Parameters
    ----------
    x : numpy.ndarray
        Input data of shape `(batch_size, in_width, in_channels)`
    w : numpy.ndarray
        Weights array of shape `(filter_width, in_channels, out_channels)`
    stride : int, optional
        The number of entries by which the filter is moved at each step.
        Defaults to 1.
    b : numpy.ndarray, optional
        Bias array to be added to the output data. Defaults to None.

    Returns
    -------
    numpy.ndarray
        Array output of the convolution step with shape
        `(batch_size, out_width, out_channels)`.

    Notes
    -----
    More details and documentation on the TensorFlow conv1d function that this
    function mimics can be found at
    https://www.tensorflow.org/api_docs/python/tf/nn/conv1d.
    """
    assert type(stride) is int, "Integer strides only!"

    # Getting some layer variables for use so we don't have to keep getting them
    n_in = x.shape[1]
    nfilters = w.shape[-1]
    k = w.shape[0]
    n_out = int((n_in - k) / stride + 1)

    i = 0
    j = 0
    # This array is size
    # (batch size, number of convolved datapoints, number of output filters)
    result = np.zeros((x.shape[0], n_out, nfilters))
    # Loop over strides keeping track of j as the index in the
    # result array.
    while i < (n_in - k + 1):
        # Collapse the last two dimensions, that is collapse
        # (batch size, kernel size, input dimension)
        # to
        # (batch size, kernel size * input dimension)
        x1 = np.reshape(x[:, i:i+k, :], (x.shape[0], -1))
        # Collapse the weights array from
        # (kernel size, input dimension, output dimension)
        # to
        # (kernel size * input dimension, output dimension)
        w1 = w.reshape((-1, nfilters))
        # Dot product for this stage of the convolution
        # Output is
        # (batch_size, output dimension)
        y = np.dot(x1, w1)

        # Store the output for returning
        result[:, j] = y
        i += stride
        j += 1

    # Make the bias vector if one wasn't passed
    if b is None:
        b = np.zeros(nfilters)

    return result + b # Adding bias

# Flattening layer, need to only flatten along dimensions
# that are not the batch size dimension.
def flatten(x):
    """Flatten an array of data.

    This function flattens an array in "C" like order, which is identical
    to the process that TensorFlow's flatten function performs. Batch size is
    not affected

    Parameters
    ----------
    x : numpy.ndarray
        Input data.

    Returns
    -------
    numpy.ndarray
        Flattened array.
    """
    return x.reshape((x.shape[0], -1), order="C")