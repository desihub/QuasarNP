import pathlib
import unittest

import numpy as np

import quasarnp.io


class TestLoadingModel(unittest.TestCase):
    def test_load_file(self):
        # Get the location of this test script and load the test_weights file
        # in this lower level directory.
        loc = pathlib.Path(__file__).parent.resolve() / "test_weights.h5"

        # If it fails here we have a problem lol.
        weights_dict = quasarnp.io.load_file(loc)

        # Check that we got all the keys we need.
        expected_keys = ['batch_normalization_1', 'batch_normalization_2',
                         'batch_normalization_3', 'batch_normalization_4',
                         'batch_normalization_5', 'conv_1', 'conv_2', 'conv_3',
                         'conv_4', 'fc_box_0', 'fc_box_1', 'fc_box_2',
                         'fc_box_3', 'fc_box_4', 'fc_box_5', 'fc_box_6',
                         'fc_common', 'fc_offset_0', 'fc_offset_1',
                         'fc_offset_2', 'fc_offset_3', 'fc_offset_4',
                         'fc_offset_5', 'fc_offset_6', 'lambda']
        # Convert to set since the dict is unordered and keys may not be in the
        # same order as the expected. Sets are unordered.
        self.assertEqual(set(weights_dict.keys()), set(expected_keys))

        # We're not going to test every single field, it's too long and messy.
        expected = [0.22520675, 0.134974, 0.25075355, 0.16675548, 0.4055901,
                    0.27923083, 0.4381064, 0.16029838, 0.19486924, 0.06135193,
                    0.23235527, 0.2077408, 0.17093728, 0.20878166, 0.41499925,
                    0.27803433, 0.09967152, 0.08413298, 0.21532041, 0.27307165,
                    0.18110749, 0.25184178, 0.92681605, 0.14645407, 0.06910174]
        observed = weights_dict["batch_normalization_1"]["moving_variance"]
        self.assertTrue(np.allclose(observed, expected))

        expected = [-0.0020517984, 0.0020787143, -0.009440172, 0.01049469,
                    -6.7694037e-4, 0.0015793968, 4.929955e-4, 0.0013135185,
                    -5.1308057e-4, 0.0051365853, -0.0023117359, 0.0031287833,
                    -6.6032336e-4, 0.0019965866, -8.7494496e-4, 0.0016014961,
                    -0.0041215573, -0.0067474414, -0.005404425, -0.008843838,
                    0.013283809, -0.0022002093, 0.0012774523, 0.0021194445,
                    -0.0034738255]
        observed = weights_dict["conv_1"]["bias"]
        self.assertTrue(np.allclose(observed, expected))

        expected = [-0.11984883, -0.16872679, -0.20790875, -0.20247324,
                    -0.17853408, -0.2611192, -0.11917564, 0.10964236,
                    -0.317558, -0.09346453, -0.21203938, -0.238487, -0.19067243]
        observed = weights_dict["fc_box_2"]["bias"]
        self.assertTrue(np.allclose(observed, expected))


if __name__ == '__main__':
    unittest.main()
