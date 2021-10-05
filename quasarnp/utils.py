"""A module containing facilities for rebinning data and processing QuasarNP output.

There are three methods in this module, one that will process the output of
QuasarNP into a human parsable form, and two that are used to rebin the DESI
wavelength grid into the QuasarNet wavelength grid. In addition, there is
a global dictionary that maps emission line names to their rest wavelength.
"""

import numpy as np

# Absorption wavelengths
absorber_IGM = {
    'Halpha'      : 6562.8,
    'OIII(5008)'  : 5008.24,
    'OIII(4933)'  : 4932.68,
    'Hbeta'       : 4862.68,
    'MgI(2853)'   : 2852.96,
    'MgII(2804)'  : 2803.5324,
    'MgII(2796)'  : 2796.3511,
    'FeII(2600)'  : 2600.1724835,
    'FeII(2587)'  : 2586.6495659,
    'MnII(2577)'  : 2576.877,
    'FeII(2383)'  : 2382.7641781,
    'FeII(2374)'  : 2374.4603294,
    'FeII(2344)'  : 2344.2129601,
    'CIII(1909)'  : 1908.734,
    'AlIII(1863)' : 1862.79113,
    'AlIII(1855)' : 1854.71829,
    'AlII(1671)'  : 1670.7886,
    'FeII(1608)'  : 1608.4511,
    'CIV(1551)'   : 1550.77845,
    'CIV(eff)'    : 1549.06,
    'CIV(1548)'   : 1548.2049,
    'SiII(1527)'  : 1526.70698,
    'SiIV(1403)'  : 1402.77291,
    'SiIV(1394)'  : 1393.76018,
    'CII(1335)'   : 1334.5323,
    'SiII(1304)'  : 1304.3702,
    'OI(1302)'    : 1302.1685,
    'SiII(1260)'  : 1260.4221,
    'NV(1243)'    : 1242.804,
    'NV(1239)'    : 1238.821,
    'LYA'         : 1215.67,
    'SiIII(1207)' : 1206.500,
    'NI(1200)'    : 1200.,
    'SiII(1193)'  : 1193.2897,
    'SiII(1190)'  : 1190.4158,
    'OI(1039)'    : 1039.230,
    'OVI(1038)'   : 1037.613,
    'OVI(1032)'   : 1031.912,
    'LYB'         : 1025.72,
}

# TODO: Make these editable and expose them publicly.
# Perhaps in the same way Farr did?
l_min = np.log10(3600.)
l_max = np.log10(10000.)
dl = 1e-3
nbins = int((l_max - l_min) / dl)
wave = 10**(l_min + np.arange(nbins) * dl)


def process_preds(preds, lines, lines_bal, verbose=True):
    """Convert network output to line confidence and redshift predictions.

    Parameters
    ----------
        preds : numpy.ndarray
            Model prediction, output of `model.predict`.
        lines : list of str
            List of line names.
        lines_bal : list of str
            List of BAL line names.
        verbose : bool, optional
            Whether or not to print verbose debug output. Defaults to True.

    Returns
    -------
        c_line : numpy.ndarray
            Confidence that each line appears in the given spectra, with
            shape `(nlines, nspec)`.
        z_line : numpy.ndarray
            Estimated redshift of the spectra derived from each line, with
            shape `(nlines, nspec)`.
        zbest : numpy.ndarray
            Redshift of the most confident line for each spectra with length
            `nspec`.
        c_line_bal : numpy.ndarray
            Confidence that each BAL line appears in the given spectra, with
            shape `(nlines_bal, nspec)`.
        z_line_bal : numpy.ndarray
            Estimated redshift of the spectra derived from each BAL line, with
            shape `(nlines_bal, nspec)`.

    Notes
    -----
    This method determines the number of trained lines and BAL lines by setting
    `nlines = len(lines)` and `nlines_bal = len(lines_bal)`.
    """
    assert len(lines) + len(lines_bal) == len(preds), "Total number of lines does not match number of predictions!"

    nspec, nboxes = preds[0].shape
    nboxes //= 2
    if verbose:
        print(f"INFO: nspec = {nspec}, nboxes={nboxes}")
    nlines = len(lines)

    # Doing non BAL lines first
    c_line = np.zeros((nlines, nspec))
    z_line = np.zeros_like(c_line) # This ensures they're always the same shape.
    i_to_wave = lambda x: np.interp(x, np.arange(len(wave)), wave)

    for il, line in enumerate(lines):
        l = absorber_IGM[line]

        # j is the box number, offset is how far into the box the line is predicted to be
        j = preds[il][:, :13].argmax(axis=1)
        offset  = preds[il][np.arange(nspec, dtype=int), nboxes + j]

        # Confidence in line, redshift of line
        c_line[il] = preds[il][:, :13].max(axis=1)
        z_line[il] = (i_to_wave((j + offset) * len(wave) / nboxes) / l) - 1

    # Not "best redshift", rather "redshift of most confident line"
    zbest = z_line[c_line.argmax(axis=0), np.arange(nspec)]
    zbest = np.array(zbest)

    # Code for BAL boxes is the same as above just run on the BAL lines.
    nlines_bal = len(lines_bal)
    c_line_bal = np.zeros((nlines_bal, nspec))
    z_line_bal = np.zeros_like(c_line_bal)

    for il, line in enumerate(lines_bal):
        l = absorber_IGM[line]

        j = preds[nlines+il][:, :13].argmax(axis=1)
        offset  = preds[il+nlines][np.arange(nspec, dtype=int), nboxes + j]

        c_line_bal[il] = preds[il + nlines][:, :13].max(axis=1)
        z_line_bal[il] = (i_to_wave((j + offset) * len(wave) / nboxes) / l) - 1

    return c_line, z_line, zbest, c_line_bal, z_line_bal


def regrid(old_grid):
    """Generate the mapping from the old wavelength grid to the QuasarNet grid.

    Parameters
    ----------
    old_grid : numpy.ndarray
        The old wavelength grid.

    Returns
    -------
    bins : numpy.ndarray
        Array of length `len(old_grid)` where each element is the bin number
        in the new grid that the old wavelength bin is assigned to.
    w : numpy.ndarray
        Array of length `len(old_grid)` where each element is True if the old
        wavelength bin is contained within the new grid boundaries and False if
        it is not.
    """
    bins = np.floor((np.log10(old_grid) - l_min) / dl).astype(int)
    w = (bins >= 0) & (bins < nbins)

    return bins, w


def rebin(flux, ivar, w_grid):
    """Rebin flux to the QuasarNet wavelength grid.

    The process for rebinning flux is as follows. First, the flux is multiplied
    by the ivar. Then, each bin from the old wavelength grid is assigned to
    a new bin on the new grid. The `flux*ivar` assigned to each new bin
    is summed and stored. The `ivar` assigned to each bin is also summed and
    stored. These rebinned values are then returned.

    Parameters
    ----------
    flux : numpy.ndarray
        Input flux array of shape `(nspec, len(w_grid))`.
    ivar : numpy.ndarray
        Input ivar array of shape `(nspec, len(w_grid))`.
    w_grid: numpy.ndarray
        Input wavelength grid.

    Returns
    -------
    flux_out : numpy.ndarray
        Input flux rebinned onto the QuasarNet wavelength grid.
    ivar_out : numpy.ndarray
        Input ivar rebinned onto the QuasarNet wavelength grid.

    See Also
    --------
    regrid : Function that converts the old wavelength grid to the new grid.
    """
    new_grid, w = regrid(w_grid)

    fl_iv = flux * ivar

    # len(flux) will give number of spectra,
    # len(new_grid) will give number of output bins
    flux_out = np.zeros((len(flux), nbins))
    ivar_out = np.zeros_like(flux_out)

    # These lines are necessary for SDSS spectra. For DESI
    # spectra nothing will change here, since the entire DESI grid is contained
    # within the QuasarNET one, but for BOSS/eBOSS the grid can extend out
    # past the QuasarNET grid and give negative bin values. I have tests that
    # confirm this still works on DESI data, don't worry.
    fl_iv = fl_iv[:, w]
    new_grid = new_grid[w]
    ivar_temp = ivar[:, w]

    for i in range(len(flux)):
        c = np.bincount(new_grid, weights=fl_iv[i, :])
        flux_out[i, :len(c)] += c
        c = np.bincount(new_grid, weights=ivar_temp[i, :])
        ivar_out[i, :len(c)] += c

    return flux_out, ivar_out


def renormalize(flux, ivar):
    """Renormalize the flux for processing with QuasarNet.

    The process for renormalizing flux is as follows. First, the weighted mean
    of the flux is calculated, with the ivar as weights. Then the weighted root
    mean squared value of the flux minus the mean is computed, once again using
    the ivar as weights. The renormalized flux is the initial flux minus the
    weighted mean and then divided by the rms value.

    Parameters
    ----------
    flux : numpy.ndarray
        Input flux array of shape `(nspec, len(w_grid))`.
    ivar : numpy.ndarray
        Input ivar array of shape `(nspec, len(w_grid))`.

    Returns
    -------
    flux_out : numpy.ndarray
        Input flux renormalized for use with QuasarNet.
    """

    # axis=1 corresponds to the rebinned spectral axis
    # Finding the weighted mean both for normalization and for the rms
    mean = np.average(flux, axis=1, weights=ivar)[:, None]
    rms = np.sqrt(np.average((flux - mean) ** 2, axis=1, weights=ivar))[:, None]

    # Normalize by subtracting the weighted mean and dividing by the rms
    # as prescribed in the original QuasarNet paper.
    return (flux - mean) / rms
