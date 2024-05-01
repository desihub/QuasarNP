import ast
import pathlib
import unittest

import numpy as np

import fitsio

from quasarnp.io import load_desi_coadd, load_model
from quasarnp.utils import process_preds

file_loc = pathlib.Path(__file__).parent.resolve() / "test_files"

lines = ['LYA', 'CIV(1548)', 'CIII(1909)', 'MgII(2796)', 'Hbeta', 'Halpha']
lines_bal = ['CIV(1548)']

class TestModels(unittest.TestCase):
    # These tests are end-to-end tests, which means they use most if not
    # all necessary functions for QuasarNP to work. If these tests are failing in
    # addition to other tests, you should fix those ones first before tackling this one
    # in order to narrow down the problem.

    def test_6_layer(self):
        data_loc = file_loc / "test_coadd.fits"
        weights_loc = file_loc / "test_weights_6_layer.h5"

        X, w = load_desi_coadd(data_loc)

        qnp_model, _ = load_model(weights_loc)
        qnp_predict = qnp_model.predict(X[:, :, None])

        qnet_predict = np.load(file_loc / "qnet_output_6_layer.npy")

        # We can use a pretty liberal tolerance on this since the two can be pretty different
        # in some cases since TF and NP handle values close to 0 differently. What's more
        # important is when we process the predictions (next test) that they come out
        # the same.
        self.assertTrue(np.allclose(qnp_predict, qnet_predict, atol=0.1))

        qnp_process = process_preds(qnp_predict, lines, lines_bal)
        qnet_process = process_preds(qnet_predict, lines, lines_bal)

        # Zeroth index is the confidences, 3rd index is the BAL confidences
        self.assertTrue(np.allclose(qnp_process[0], qnet_process[0]))
        self.assertTrue(np.allclose(qnp_process[3], qnet_process[3]))


if __name__ == '__main__':
    unittest.main()
