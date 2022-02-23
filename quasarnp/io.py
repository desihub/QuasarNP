"""A module containing facilities for loading data and QuasarNP models.

Methods are provided for separately loading a QuasarNP model from a weights
file as well as loading DESI data, both by exposure number and by directory.
A method is also provided to load a DESI coadd file. Two legacy methods are
provided to interop with SDSS data: one to load a truth table and one to
load a SDSS data file.
"""
import json
from pathlib import Path

import fitsio
import h5py
import numpy as np

from .model import QuasarNP
from .utils import rebin, renormalize


def load_file(filename):
    """Load a weights file as a dictionary.

    Parameters
    ----------
    filename : str
        The name of the weights file.
    Returns
    -------
    dict
        Dictionary that maps layer names to layer weights.
    """
    result = {}

    with h5py.File(filename, "r") as f:
        m_weights = f['model_weights']
        m_config = json.loads(f.attrs["model_config"])

        # Some versions of TF/Keras are 1 indexed and so bn layers start
        # at batch_normalization_1. Some versions are 0 indexed and start at
        # batch_normalization. Former is easer to account for in
        # model object since it is of consistent form. Thus we modify names
        # to match that syntax.
        inc = False
        if "batch_normalization" in m_weights:
            inc = True

        for k1, v1 in m_weights.items():
            if len(v1) == 0:
                if k1 == "lambda":
                    result[k1] = []
                continue
            a = v1[k1]

            data_dict = {}
            for k2, v2 in a.items():
                # This :-2 strips off the :0 at the end of weights names.
                data_dict[k2[:-2]] = v2[()]

            # Handles the 0 vs 1 indexed batch_norm (see note above)
            name = k1
            if "batch_normalization" in name:
                if name == "batch_normalization":
                    name = "batch_normalization_1"
                elif inc:
                    name = name[:-1] + str(int(name[-1:]) + 1)

            result[name] = data_dict

    names = [m_config["config"]["layers"][i]["config"]["name"] for i in range(len(m_config["config"]["layers"]))]
    conv_layers = [k for k in result.keys() if k.startswith("conv")]

    config_dict = {}
    for l in conv_layers:
        idx = names.index(l)
        temp = {}
        for k in ["padding", "strides"]:
            temp[k] = m_config["config"]["layers"][idx]["config"][k]
        config_dict[l] = temp

    return result, config_dict


def load_model(filename):
    """Load a weights file and return a callable model object.

    Parameters
    ----------
    filename : str
        The name of the weights file.
    Returns
    -------
    QuasarNP
        Callable QuasarNP model with the weights provided by `filename`.
    """
    db, config = load_file(filename)

    nlayers = len([k for k in db.keys() if k.startswith("conv")])

    if "lambda" in db:
        return QuasarNP(db, rescale=True, nlayers=nlayers, config_dict=config)
    else:
        return QuasarNP(db, nlayers=nlayers, config_dict=config)


def read_truth(fi):
    """ Read a list of truth files and return a dictionary of truth values.

    This is a legacy function ported from QuasarNet, and is designed to load
    SDSS data files to generate a truth table.

    Parameters
    ----------
    fi : list of str
        List of file names of truth files.
    Returns
    -------
    dict
        Dictionary that maps `thing_id` to truth metadata.
    """

    class metadata:
        pass

    cols = ['Z_VI', 'PLATE', 'MJD', 'FIBERID', 'CLASS_PERSON',
            'Z_CONF_PERSON', 'BAL_FLAG_VI', 'BI_CIV']

    truth = {}

    for f in fi:
        h = fitsio.FITS(f)
        tids = h[1]['THING_ID'][:]
        cols_dict = {c.lower(): h[1][c][:] for c in cols}
        h.close()
        for i, t in enumerate(tids):
            m = metadata()
            for c in cols_dict:
                setattr(m, c, cols_dict[c][i])
            truth[t] = m

    return truth


def read_data(fi, truth=None, z_lim=2.1, return_pmf=False, nspec=None):
    """Read data from input file.

    This is a legacy function ported from QuasarNet, and is designed to load
    SDSS data files.
    Returns a tuple containing (tids, X, Y, z, bal) if `return_pmf` is `False`,
    otherwise returns a tuple containing (tids, X, Y, z, bal, plate, mjd, fid).

    Parameters
    ----------
    fi : list of str
        List of data files to load.
    truth : dict, optional
        Dictionary that maps `thing_id`` to truth metadata.
    z_lim : float, optional
        Redshift to use when applying a z-cut. Defaults to 2.1.
    return_pmf : bool, optional
        Whether or not to return the `plate`, `mjd` and `fiberid`.
        Defaults to False.
    nspec : int, optional
        Number of spectra to read. Defaults to None (all spectra)

    Returns
    -------
    tids : list of float
        A list of `thing_id`.
    X : numpy.ndarray
        Renormalized and rebinned spectra.
    Y : numpy.ndarray
        Classification vector of shape (`nqso`, 5) with the following entries:
                        STAR = (1,0,0,0,0), GAL = (0,1,0,0,0)
                        QSO_LZ = (0,0,1,0,0), QSO_HZ = (0,0,0,1,0)
                        BAD = (0,0,0,0,1)
    z : numpy.ndarray
        Array of redshifts.
    bal : numpy.ndarray
        Truth array indicating whether each QSO is a BAL QSO or not. Each
        element is set to `1` if True or `0` if False.
    plate : numpy.ndarray
        Array of plate ids. Only returned when `return_pmf` is True.
    mjd : numpy.ndarray
        Array of mean julien dates. Only returned when `return_pmf` is True.
    fid : float
        Array of fiber ids. Only returned when `return_pmf` is True.
    """

    tids = []
    X = []
    Y = []
    z = []
    bal = []

    if return_pmf:
        plate = []
        mjd = []
        fid = []

    for f in fi:
        print('INFO: reading data from {}'.format(f))
        h = fitsio.FITS(f)
        if nspec is None:
            nspec = h[1].get_nrows()
        aux_tids = h[1]['TARGETID'][:nspec].astype(int)
        # Remove thing_id == -1 or not in sdrq
        w = (aux_tids != -1)
        if truth is not None:
            w_in_truth = np.in1d(aux_tids, list(truth.keys()))
            print((f"INFO: Removing {(~w_in_truth).sum()}"
                   " spectra missing in truth"), flush=True)
            w &= w_in_truth
        aux_tids = aux_tids[w]
        aux_X = h[0][:nspec, :]

        aux_X = aux_X[w]
        if return_pmf:
            aux_plate = h[1]['PLATE'][:][w]
            aux_mjd = h[1]['MJD'][:][w]
            aux_fid = h[1]['FIBERID'][:][w]
            plate += list(aux_plate)
            mjd += list(aux_mjd)
            fid += list(aux_fid)

        X.append(aux_X)
        tids.append(aux_tids)

        print(f"INFO: Found {aux_tids.shape} spectra in file {f}")

    tids = np.concatenate(tids)
    X = np.concatenate(X)

    if return_pmf:
        plate = np.array(plate)
        mjd = np.array(mjd)
        fid = np.array(fid)

    we = X[:, 443:]
    w = we.sum(axis=1) == 0
    print("INFO: removing {} spectra with zero weights".format(w.sum()))
    X = X[~w]
    tids = tids[~w]

    if return_pmf:
        plate = plate[~w]
        mjd = mjd[~w]
        fid = fid[~w]

    mdata = np.average(X[:, :443], weights=X[:, 443:], axis=1)
    sdata = np.average((X[:, :443] - mdata[:, None])**2,
                       weights=X[:, 443:], axis=1)
    sdata = np.sqrt(sdata)

    w = sdata == 0
    print("INFO: removing {} spectra with zero flux".format(w.sum()))
    X = X[~w]
    tids = tids[~w]
    mdata = mdata[~w]
    sdata = sdata[~w]

    if return_pmf:
        plate = plate[~w]
        mjd = mjd[~w]
        fid = fid[~w]

    X = X[:, :443] - mdata[:, None]
    X /= sdata[:, None]

    if truth is None:
        if return_pmf:
            return tids, X, plate, mjd, fid
        else:
            return tids, X

    # Remove zconf == 0 (not inspected)
    observed = [(truth[t].class_person > 0) or
                (truth[t].z_conf_person > 0) for t in tids]
    observed = np.array(observed, dtype=bool)
    tids = tids[observed]
    X = X[observed]

    if return_pmf:
        plate = plate[observed]
        mjd = mjd[observed]
        fid = fid[observed]

    # Fill redshifts
    z = np.zeros(X.shape[0])
    z[:] = [truth[t].z_vi for t in tids]

    # Fill bal
    bal = np.zeros(X.shape[0])
    bal[:] = [(truth[t].bal_flag_vi * (truth[t].bi_civ > 0)) -
              (not truth[t].bal_flag_vi) * (truth[t].bi_civ == 0)
              for t in tids]

    # Fill classes
    # Classes: 0 = STAR, 1=GALAXY, 2=QSO_LZ, 3=QSO_HZ, 4=BAD (zconf != 3)
    nclasses = 5
    sdrq_class = np.array([truth[t].class_person for t in tids])
    z_conf = np.array([truth[t].z_conf_person for t in tids])

    Y = np.zeros((X.shape[0], nclasses))
    # STAR
    w = (sdrq_class == 1) & (z_conf == 3)
    Y[w, 0] = 1

    # GALAXY
    w = (sdrq_class == 4) & (z_conf == 3)
    Y[w, 1] = 1

    # QSO_LZ
    w = ((sdrq_class == 3) | (sdrq_class == 30)) & (z < z_lim) & (z_conf == 3)
    Y[w, 2] = 1

    # QSO_HZ
    w = ((sdrq_class == 3) | (sdrq_class == 30)) & (z >= z_lim) & (z_conf == 3)
    Y[w, 3] = 1

    # BAD
    w = z_conf != 3
    Y[w, 4] = 1

    # Check that all spectra have exactly one classification
    assert (Y.sum(axis=1).min() == 1) and (Y.sum(axis=1).max() == 1)

    if return_pmf:
        return tids, X, Y, z, bal, plate, mjd, fid

    return tids, X, Y, z, bal


# DESI Related IO below this point
###############################################################################

def load_desi_exposure(dir_name, spec_number,
                       fibers=np.ones(500, dtype="bool")):
    """Load and renormalize a raw DESI spectrographic exposure.

    This method will load B, R and Z cframe files in sequence. First, spectra
    are rebinned to the QuasarNet wavelength grid. Rebinned spectra are divided
    by the rebinned IVAR to reweight the spectra. Next, rebinned spectra are
    normalized by subtracting the weighted mean of the spectra and then
    dividing the resultant spectra by its weighted rms. The rebinned IVAR is
    used for weighting. Any spectra where the IVAR is 0 for the entire
    wavelength grid is discarded.

    Parameters
    ----------
    dir_name : str
        Directory to load exposure from.
    spec_number : int
        Spectrograph number to load.
    fibers : numpy.ndarray, optional.
        Array of length 500 indicating whether each fiber should be loaded.
        True if the fiber should be loaded, False otherwise. Defaults to
        True for all 500 fibers.

    Returns
    -------
    X_out : numpy.ndarray
        Renormalized and rebinned spectra. Output spectra will have shape
        `(nspectra, nbins)` where `nbins=443` for the QuasarNet wavelength
        grid.
    w : numpy.ndarray
        Array of length `sum(fibers == True)` where each element is True if
        the spectra was kept in `X_out` and False if the spectra was discarded.

    See Also
    ---------
    load_desi_daily : Load a daily exposure.
    """
    assert len(fibers) == 500, ("fibers input must include True/False"
                                " for all 500 fibers.")
    assert 0 <= spec_number and spec_number <= 9, ("spec_number must be"
                                                   " between 0 and 9.")

    file_loc = Path(dir_name)
    exp_id = file_loc.parts[-1]

    # Load each cam sequentially, then rebin and merge
    # We will be rebinning down to 443, which is the input size of QuasarNet
    nfibers = np.sum(fibers > 0)
    X_out = np.zeros((nfibers, 443))

    # ivar_out is the weights out, i.e. the ivar, we use this for normalization
    # Use zeros_like so we only have to change one
    ivar_out = np.zeros_like(X_out)

    cams = ["B", "R", "Z"]
    for c in cams:
        im_path = file_loc / f"cframe-{c.lower()}{spec_number}-{exp_id}.fits"
        with fitsio.FITS(im_path) as h:
            # Load the flux and ivar
            flux = h["FLUX"].read()[fibers, :]
            ivar = h["IVAR"].read()[fibers, :]
            w_grid = h["WAVELENGTH"].read()

        # Rebin the flux and ivar
        new_flux, new_ivar = rebin(flux, ivar, w_grid)

        X_out += new_flux
        ivar_out += new_ivar

    non_zero = ivar_out != 0
    X_out[non_zero] /= ivar_out[non_zero]

    nonzero_weights = np.sum(ivar_out, axis=1) != 0
    print(f"{nfibers - np.sum(nonzero_weights)} spectra with zero weights")
    X_out = X_out[nonzero_weights]
    ivar_out = ivar_out[nonzero_weights]

    X_out = renormalize(X_out, ivar_out)
    return X_out, np.where(nonzero_weights)[0]


def load_desi_coadd(filename, rows=None):
    """Load and renormalize a DESI coadded spectrographic exposure.

    This method will load a coadd file and renormalize as follows. First,
    spectra are rebinned to the QuasarNet wavelength grid. Rebinned spectra
    are divided by the rebinned IVAR to reweight the spectra. Next, rebinned
    spectra are normalized by subtracting the weighted mean of the spectra and
    then dividing the resultant spectra by its weighted rms. The rebinned IVAR
    is used for weighting. Any spectra where the IVAR is 0 for the entire
    wavelength grid is discarded.

    Parameters
    ----------
    filename : str
        Full path and filename of the coadd file to load.
    rows : numpy.ndarray, optional.
        Boolean array indicating whether each row should be loaded. True
        if the row should be loaded, False otherwise. Defaults to None, which
        loads all rows.

    Returns
    -------
    X_out : numpy.ndarray
        Renormalized and rebinned spectra. Output spectra will have shape
        `(nspectra, nbins)` where `nbins=443` for the QuasarNet wavelength grid.
    w : numpy.ndarray
        Array of length `sum(rows == True)` where each element is True if
        the spectra was kept in `X_out` and False if the spectra was discarded.

    See Also
    --------
    load_desi_daily : Load a daily exposure.
    """
    cams = ["B", "R", "Z"]
    with fitsio.FITS(filename) as h:
        # Load each cam sequentially, then rebin and merge
        # We will be rebinning down to 443, the input size of QuasarNet
        if rows is None:
            nfibers = len(h['B_FLUX'].read())
            rows = np.ones(nfibers, dtype='bool')
        else:
            nfibers = np.sum(rows > 0)
        X_out = np.zeros((nfibers, 443))
        # ivar_out is the weights out, we use this for normalization
        # Use zeros_like so we only have to change one
        ivar_out = np.zeros_like(X_out)
        for c in cams:
            fluxname = f"{c}_FLUX"
            ivarname = f"{c}_IVAR"
            wname = f"{c}_WAVELENGTH"

            # Load the flux and ivar
            flux = h[fluxname].read()[rows, :]
            ivar = h[ivarname].read()[rows, :]
            w_grid = h[wname].read()

            # Rebin the flux and ivar
            new_flux, new_ivar = rebin(flux, ivar, w_grid)

            X_out += new_flux
            ivar_out += new_ivar

    non_zero = ivar_out != 0
    X_out[non_zero] /= ivar_out[non_zero]

    nonzero_weights = np.sum(ivar_out, axis=1) != 0

    # f"{nfibers - np.sum(nonzero_weights)} spectra with zero weights"
    X_out = X_out[nonzero_weights]
    ivar_out = ivar_out[nonzero_weights]

    X_out = renormalize(X_out, ivar_out)
    return X_out, np.where(nonzero_weights)[0]


def load_desi_daily(night, exp_id, spec_number,
                    fibers=np.ones(500, dtype="bool")):
    """Load and renormalize a daily DESI spectrographic exposure.

    This method will load B, R and Z cframe files in sequence. First, spectra
    are rebinned to the QuasarNet wavelength grid. Rebinned spectra are
    divided by the rebinned IVAR to reweight the spectra. Next, rebinned
    spectra are normalized by subtracting the weighted mean of the spectra and
    then dividing the resultant spectra by its weighted rms. The rebinned IVAR
    is used for weighting. Any spectra where the IVAR is 0 for the entire
    wavelength grid is discarded.

    Parameters
    ----------
    night : int or str
        Night on which the exposure was taken.
    exp_id : str
        Exposure ID of the exposure.
    spec_number : int
        Spectrograph number to load.
    fibers : numpy.ndarray, optional.
        Array of length 500 indicating whether each fiber should be loaded.
        True if the fiber should be loaded, False otherwise.
        Defaults to True for all 500 fibers.

    Returns
    -------
    X_out : numpy.ndarray
        Renormalized and rebinned spectra. Output spectra will have shape
        `(nspectra, nbins)` where `nbins=443` for the QuasarNet wavelength
        grid.
    w : numpy.ndarray
        Array of length `sum(fibers == True)` where each element is True if
        the spectra was kept in `X_out` and False if the spectra was discarded.

    See Also
    --------
    load_desi_exposure : Used by load_desi_daily to load the given exposure.
    load_desi_coadd : Load a coadded exposure.
    """
    assert len(fibers) == 500, ("fibers input must include True/False"
                                " for all 500 fibers.")
    assert 0 <= spec_number and spec_number <= 9, ("spec_number must be"
                                                   " between 0 and 9.")

    # For now load daily cframes files
    # TODO: add support for loading arbitrary cframes.
    # TODO: Add support for loading by tile id + e rather than date + e
    root = "/global/cfs/cdirs/desi/spectro/redux/daily/exposures"
    file_loc = Path(root, night, exp_id)

    return load_desi_exposure(file_loc, spec_number, fibers)


# BOSS Related IO below this point
###############################################################################

def read_spall(file_loc):
    """ Read metadata from a spAll file.

    Parameters
    ----------
        file_loc : string or Path
            Full path and filename of the spAll file to read.
    Returns
    -------
        tid : numpy.ndarray
            Array of integer THING_IDs.
        pmf2tid : dict
            Dictionary mapping (PLATE, MJD, FIBERID) to THING_ID.

    """
    # Open the file, and read plate, mjd, fiberid, thing_id, specprimary.
    # read() is faster than using [:] since we can read all the columns at once.
    with fitsio.FITS(file_loc) as h:
        d = h[1].read(columns=["PLATE", "MJD", "FIBERID", "THING_ID",
                               "SPECPRIMARY"])

    # Need to cast THING_ID to int and it's easier to read to do it here.
    tid = d["THING_ID"].astype(int)
    pmf2tid = {(p, m, f): t for p, m, f, t, s in zip(d["PLATE"], d["MJD"],
                                                     d["FIBERID"], tid,
                                                     d["SPECPRIMARY"])}

    return tid, pmf2tid
