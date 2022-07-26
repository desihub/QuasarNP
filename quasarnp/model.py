"""A module defining the QuasarNP model object.

QuasarNP is a pure numpy implementation of the original TensorFlow
QuasarNet. This module only defines the QuasarNP model object and provides
no other functionality on its own.
"""
import numpy as np

from .layers import conv1d, batch_normalization, dense, flatten, relu, sigmoid, linear

class QuasarNP():
    """A QuasarNP model object. The model holds all of the relevant layer
    information, including layer names, weights and definitions.

    Parameters
    ----------
    weights : dict
        Dictionary that maps layer names to layer weights.
    nlines : int, optional
        Number of lines this model was trained for. Defaults to 7.
    rescale : bool, optional
        Whether or not to rescale the output of the box layers. Defaults to False.
    """
    def __init__(self, weights, nlines=7, rescale=False, nlayers=4, config_dict=None):
        # Store the weights to access later.
        self.weights = weights
        self.config = config_dict
        self.nlines = nlines
        self.apply_rescale = rescale

        self.convs = []
        for i in range(nlayers):
            n_conv = f"conv_{i + 1}"
            self.convs.append(n_conv)

        # Don't forget to flatten here before dense layer
        self.dense = lambda x: dense(x, weights["fc_common"]["kernel"], weights["fc_common"]["bias"], linear)

        w_batch = self.weights[f"batch_normalization_{nlayers + 1}"]
        self.batch_norm = lambda x: batch_normalization(x, w_batch["moving_mean"], w_batch["moving_variance"] ,
                                                             w_batch["beta"], w_batch["gamma"], 0.001)
        # Do not forget that there is a relu here after the batch_norm
        # Then boxes, this is a rescale function for the box offset
        self.rescale = lambda x: -0.1 + 1.2 * x


    def conv_layer(self, x, name):
        w_conv = self.weights[name]
        # Gets the number name of this conv layer for the batch norm
        n_batch = f"batch_normalization_{name[-1]}"
        w_batch = self.weights[n_batch]

        if self.config is not None:
            stride = self.config[name]["strides"][0]
            padding = self.config[name]["padding"]

            y = conv1d(x, w_conv["kernel"], stride=stride, b=w_conv["bias"],
                       padding=padding)
        else:
            y = conv1d(x, w_conv["kernel"], stride=2, b=w_conv["bias"])
        # Default epsilon in Tensorflow is 0.001
        # QuasarNet does not seem to change this value so I will use this for now.
        y = batch_normalization(y, w_batch["moving_mean"], w_batch["moving_variance"] ,
                                w_batch["beta"], w_batch["gamma"], 0.001)
        return relu(y)

    def predict(self, x_input):
        """Run a set of spectra and generate predictions.

        Parameters
        ----------
        x_input : numpy.ndarray
            Input array of spectra, with shape `(nspectra, nbins, 1)`

        Returns
        -------
        list of numpy.ndarray
            Moodel predicted values per trained line, with length  `nlines`.

        Notes
        -----
        In order to mimic QuasarNet behaviour in QuasarNP, `predict` does not
        directly take as input the output of `load_desi_exposure` or
        `load_desi_daily`. You must first expand the dimensions of the loaded
        data before loading. For example:

        >>> data, w = load_desi_daily("20210107", "00071246", 1)
        >>> data = data[:, :, None]
        >>> model.predict(data)
        """
        x_output = np.copy(x_input) # To avoid side effects
        for name in self.convs:
            x_output = self.conv_layer(x_output, name)

        x_output = flatten(x_output)
        x_output = relu(self.batch_norm(self.dense(x_output)))

        outputs = []
        for i in range(self.nlines):
            w_box = self.weights[f"fc_box_{i}"]
            box = dense(x_output, w_box["kernel"], w_box["bias"], sigmoid)

            w_offset = self.weights[f"fc_offset_{i}"]
            offset = dense(x_output, w_offset["kernel"], w_offset["bias"], sigmoid)
            if self.apply_rescale: offset = self.rescale(offset)

            o = np.concatenate([box, offset], axis=1)
            outputs.append(o)

        return outputs

    # This definition is here to mimic Tensorflow/QuasarNet behaviour.
    def __call__(self, x_input):
        return self.predict(x_input)