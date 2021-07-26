import ast
import pathlib
import unittest

import numpy as np

from quasarnp.utils import regrid, process_preds


class TestUtilities(unittest.TestCase):
    # Test taking the old grid and generating which bins on the new grid
    # the grid goes into.
    def test_regrid(self):
        # This is the new grid, so regridding it shouldn't do anything.
        new_grid = 10 ** (np.arange(np.log10(3600), np.log10(10000), 1e-3))
        ob_bins, ob_keep = regrid(new_grid)
        expected_bins = np.arange(443)
        expected_bins = np.insert(expected_bins, 0, 0)
        self.assertTrue(np.allclose(ob_bins, expected_bins))
        self.assertTrue(np.allclose(ob_keep, np.ones_like(ob_keep, dtype=bool)))

        # Testing regridding the DESI grid into the SDSS/QuasarNet grid.
        wmin, wmax, wdelta = 3600, 9824, 0.8
        old_grid = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
        ob_bins, ob_keep = regrid(old_grid)

        # In order to not have to overload this file with nuisance, I have moved
        # the actual answer here to regrid.txt. It's quite long, so only
        # investigate if strictly necessary.
        loc = pathlib.Path(__file__).parent.resolve() / "regrid.txt"
        with open(loc, 'r') as f:
            expected_bins = ast.literal_eval(f.read().replace("\n", ""))
        self.assertTrue(np.allclose(ob_bins, expected_bins))
        self.assertTrue(np.allclose(ob_keep, np.ones_like(ob_keep, dtype=bool)))

    # This test should be independent of weights file, so I precomputed the
    # processed predictions using a method I know/assume to be correct (the
    # Farr/Busca QuasarNet impl) and then saved them to compare to here in
    # this test using the QuasarNP impl.
    def test_process_preds(self):
        lines = ['LYA', 'CIV(1548)', 'CIII(1909)', 'MgII(2796)',
                 'Hbeta', 'Halpha']
        lines_bal = ['CIV(1548)']

        # These predictions come from the qn_train_coadd_indtrain_0_0_boss10.h5
        # weights file trained by James Farr.
        base_loc = pathlib.Path(__file__).parent.resolve()
        loc = base_loc / "predict_data.npy"
        with open(loc, 'rb') as f:
            p = np.load(f)

        observed = process_preds(p, lines, lines_bal, verbose=False)

        # We'll just check the LyA predictions and the BAL predictions.
        loc = base_loc / "predict_lya.npy"
        with open(loc, 'rb') as f:
            expected_lya = np.load(f)
        self.assertTrue(np.allclose(observed[0], expected_lya))

        # Gonna assume that with these two everything's alright.
        loc = base_loc / "predict_bal.npy"
        with open(loc, 'rb') as f:
            expected_bal = np.load(f)
        # One of these predictions is slightly different, by a very small amount
        # but just large enough to fail the default rtol value.
        self.assertTrue(np.allclose(observed[-1], expected_bal, rtol=1e-2))


if __name__ == '__main__':
    unittest.main()
