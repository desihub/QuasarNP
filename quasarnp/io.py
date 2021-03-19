from pathlib import Path

import fitsio
import h5py
import numpy as np

from .model import QuasarNP
from .utils import rebin

def load_file(filename):
    result = {}

    with h5py.File(filename, "r") as f:
        m_weights = f['model_weights']

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

    print(result.keys())
    return result

def load_model(filename):
    db = load_file(filename)
    if "lambda" in db:
        return QuasarNP(db, rescale=True)
    else:
        return QuasarNP(db)


def read_truth(fi):
    '''
    reads a list of truth files and returns a truth dictionary
    Arguments:
        fi -- list of truth files (list of string)
    Returns:
        truth -- dictionary of THING_ID: metadata instance
    '''

    class metadata:
        pass

    cols = ['Z_VI','PLATE',
            'MJD','FIBERID','CLASS_PERSON',
            'Z_CONF_PERSON','BAL_FLAG_VI','BI_CIV']

    truth = {}

    for f in fi:
        h = fitsio.FITS(f)
        tids = h[1]['THING_ID'][:]
        cols_dict = {c.lower():h[1][c][:] for c in cols}
        h.close()
        for i,t in enumerate(tids):
            m = metadata()
            for c in cols_dict:
                setattr(m,c,cols_dict[c][i])
            truth[t] = m

    return truth

def read_data(fi, truth=None, z_lim=2.1,
        return_pmf=False, nspec=None):
    '''
    reads data from input file
    Arguments:
        fi -- list of data files (string iterable)
        truth -- dictionary thind_id => metadata
        z_lim -- hiz/loz cut (float)
        return_pmf -- if True also return plate,mjd,fiberid
        nspec -- read this many spectra
    Returns:
        tids -- list of thing_ids
        X -- spectra reformatted to be fed to the network (numpy array)
        Y -- truth vector (nqso, 5):
                           STAR = (1,0,0,0,0), GAL = (0,1,0,0,0)
                           QSO_LZ = (0,0,1,0,0), QSO_HZ = (0,0,0,1,0)
                           BAD = (0,0,0,0,1)
        z -- redshift (numpy array)
        bal -- 1 if bal, 0 if not (numpy array)
    '''

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
        h=fitsio.FITS(f)
        if nspec is None:
            nspec = h[1].get_nrows()
        aux_tids = h[1]['TARGETID'][:nspec].astype(int)
        ## remove thing_id == -1 or not in sdrq
        w = (aux_tids != -1)
        if truth is not None:
            w_in_truth = np.in1d(aux_tids, list(truth.keys()))
            print("INFO: removing {} spectra missing in truth".format((~w_in_truth).sum()),flush=True)
            w &= w_in_truth
        aux_tids = aux_tids[w]
        aux_X = h[0][:nspec,:]

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

        print("INFO: found {} spectra in file {}".format(aux_tids.shape, f))

    tids = np.concatenate(tids)
    X = np.concatenate(X)

    if return_pmf:
        plate = np.array(plate)
        mjd = np.array(mjd)
        fid = np.array(fid)

    we = X[:,443:]
    w = we.sum(axis=1)==0
    print("INFO: removing {} spectra with zero weights".format(w.sum()))
    X = X[~w]
    tids = tids[~w]

    if return_pmf:
        plate = plate[~w]
        mjd = mjd[~w]
        fid = fid[~w]

    mdata = np.average(X[:,:443], weights = X[:,443:], axis=1)
    sdata = np.average((X[:,:443]-mdata[:,None])**2,
            weights = X[:,443:], axis=1)
    sdata=np.sqrt(sdata)

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

    X = X[:,:443]-mdata[:,None]
    X /= sdata[:,None]

    if truth==None:
        if return_pmf:
            return tids,X,plate,mjd,fid
        else:
            return tids,X

    ## remove zconf == 0 (not inspected)
    observed = [(truth[t].class_person>0) or (truth[t].z_conf_person>0) for t in tids]
    observed = np.array(observed, dtype=bool)
    tids = tids[observed]
    X = X[observed]

    if return_pmf:
        plate = plate[observed]
        mjd = mjd[observed]
        fid = fid[observed]

    ## fill redshifts
    z = np.zeros(X.shape[0])
    z[:] = [truth[t].z_vi for t in tids]

    ## fill bal
    bal = np.zeros(X.shape[0])
    bal[:] = [(truth[t].bal_flag_vi*(truth[t].bi_civ>0))-\
            (not truth[t].bal_flag_vi)*(truth[t].bi_civ==0) for t in tids]

    ## fill classes
    ## classes: 0 = STAR, 1=GALAXY, 2=QSO_LZ, 3=QSO_HZ, 4=BAD (zconf != 3)
    nclasses = 5
    sdrq_class = np.array([truth[t].class_person for t in tids])
    z_conf = np.array([truth[t].z_conf_person for t in tids])

    Y = np.zeros((X.shape[0],nclasses))
    ## STAR
    w = (sdrq_class==1) & (z_conf==3)
    Y[w,0] = 1

    ## GALAXY
    w = (sdrq_class==4) & (z_conf==3)
    Y[w,1] = 1

    ## QSO_LZ
    w = ((sdrq_class==3) | (sdrq_class==30)) & (z<z_lim) & (z_conf==3)
    Y[w,2] = 1

    ## QSO_HZ
    w = ((sdrq_class==3) | (sdrq_class==30)) & (z>=z_lim) & (z_conf==3)
    Y[w,3] = 1

    ## BAD
    w = z_conf != 3
    Y[w,4] = 1

    ## check that all spectra have exactly one classification
    assert (Y.sum(axis=1).min()==1) and (Y.sum(axis=1).max()==1)

    if return_pmf:
        return tids,X,Y,z,bal,plate,mjd,fid

    return tids,X,Y,z,bal


def load_desi_exposure(night, exp_id, spec_number, fibers=np.ones(500, dtype="bool")):
    assert len(fibers) == 500, "fibers input must include True/False for all 500 fibers."
    assert 0 <= spec_number and spec_number <= 9, "spec_number must be between 0 and 9"

    # For now load cascades cframes files
    # Can/should be changed later.
    # TODO: Add support for loading by tile id + e rather than date + e
    root = "/global/cfs/cdirs/desi/spectro/redux/cascades/exposures"
    file_loc = Path(root, night, exp_id)

    # Load each cam sequentially, then rebin and merge
    # We will be rebinning down to 443, which is the input size of QuasarNet
    nfibers = np.sum(fibers)
    X_out = np.zeros((nfibers, 443))

    # ivar_out is the weights out, i.e. the ivar, we use this for normalization
    ivar_out = np.zeros_like(X_out) # Use zeros_like so we only have to change one

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

    # axis=1 corresponds to the rebinned spectral axis
    # Finding the weighted mean both for normalization and for the rms
    mean = np.average(X_out, axis=1, weights=ivar_out)[:, None]
    rms = np.sqrt(np.average((X_out - mean) ** 2 ,axis=1, weights=ivar_out))

    # Normalize by subtracting the weighted mean and dividing by the rms
    # as prescribed in the original QuasarNet paper.
    X_out = (X_out - mean) / rms[:, None]
    return X_out
