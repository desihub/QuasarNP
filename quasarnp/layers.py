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
    return phi(x @ w + b)

# This uses the "moving mean" which is compiled from the training set
# rather than just taking the mean of the input data like I would assume.
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
# This has same behavior as:
# https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
def batch_normalization(x, mean, var, beta, gamma, epsilon):
    return ((x - mean) / np.sqrt(var + epsilon)) * gamma + beta

# Not the fasted possible implementation since this does
# O(stride) more computations than necessary
# This assumes x and w are in the exact form you would pass
# them to tf.nn.conv1d for example
# This has same behavior as:
# https://www.tensorflow.org/api_docs/python/tf/nn/conv1d
def conv1d(x, w, stride=1, b=None):
    assert type(stride) is int, "Integer strides only!"

    # Getting some layer variables for use so we don't have to keep getting them
    n_in = x.shape[1]
    nfilters = w.shape[-1]
    k = w.shape[0]
    n_out = (n_in - k) / stride + 1
    out = []

    # Loop over each filter and convolve (correlate) it.
    for i in range(nfilters):
        y = np.correlate(x[0, :, 0], w[:, 0, i], mode="valid")[::stride]
        n_out = len(y)
        out.append(y)

    result = np.concatenate(out)
    result = result.reshape((1, n_out, nfilters), order="F")

    # Make the bias vector if one wasn't passed
    if b is None:
        b = np.zeros(nfilters)

    return result + b # Adding bias

# Tensorflow uses row major flattening which is the default for numpy
# as well but I extracted this to a function anyway.
def flatten(x):
    return x.reshape(1, order="C")