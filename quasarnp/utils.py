import numpy as np

# Absorbtion wavelengths
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

l_min = np.log10(3600.)
l_max = np.log10(10000.)
dl = 1e-3
nbins = int((l_max - l_min)/dl)
wave = 10**(l_min + np.arange(nbins)*dl)

def process_preds(preds, lines, lines_bal):
    '''
    Convert network predictions to c_lines, z_lines and z_best
    Arguments:
    preds: float, array
        model predictions, output of model.predict
    lines: string, array
        list of line names
    lines_bal: string, array
        list of BAL line names
    Returns:
    c_line: float, array
        line confidences, shape: (nlines, nspec)
    z_line: float, array
        line redshifts, shape: (nlines, nspec)
    zbest: float, array
        redshift of highest confidence line, shape: (nspec)
    c_line_bal: float, array
        line confidences of BAL lines, shape: (nlines_bal, nspec)
    z_line_bal: float, array
        line redshfits of BAL lines, shape: (nlines_bal, nspec)
    '''
    assert len(lines) + len(lines_bal) == len(preds), "Total number of lines does not match number of predictions!"

    nspec, nboxes = preds[0].shape
    nboxes //= 2
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
    cbest = c_line[c_line.argmax(axis=0), np.arange(nspec)]
    zbest = np.array(zbest)
    cbest = np.array(cbest)

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

    return c_line, z_line, zbest, cbest, c_line_bal, z_line_bal

def regrid(old_grid):
    bins = np.floor((np.log10(old_grid) - l_min) / dl).astype(int)
    w = (bins>=0) & (bins<nbins)

    return bins, w


def rebin(flux, ivar, w_grid):
    new_grid, w = regrid(w_grid)

    fl_iv = flux * ivar

    # len(flux) will give number of spectra,
    # len(new_grid) will give number of output bins
    flux_out = np.zeros((len(flux), nbins))
    ivar_out = np.zeros_like(flux_out)

    for i in range(len(flux)):
        c = np.bincount(new_grid, weights = fl_iv[i, :])
        flux_out[i, :len(c)] += c
        c = np.bincount(new_grid, weights = ivar[i, :])
        ivar_out[i, :len(c)] += c

    return flux_out, ivar_out
