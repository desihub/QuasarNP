# QuasarNP
QuasarNP is a pure numpy implementation of [QuasarNet](https://github.com/ngbusca/QuasarNET) that is designed to work on the default DESI environment at NERSC without any additional dependencies.

Indeed, QuasarNP requires only three things to work:
- Numpy
- hd5py
- fitsio

## Example Operation on DESI Data
QuasarNP is designed to be interoperable with prior QuasarNet code.

### Loading Weights
To begin we must load a weights file in order to be able to make predictions
on whether something is a quasar or not. This is quite simple and can be done
as follows, where I have also defined the lines that this model was trained
for:

```python
from quasarnp.io import load_model

lines = ['LYA','CIV(1548)','CIII(1909)', 'MgII(2796)','Hbeta','Halpha']
lines_bal = ['CIV(1548)']

model = load_model("/global/cfs/cdirs/desi/science/lya/qn_models/boss_dr12/qn_train_coadd_indtrain_0_0_boss10.h5")
```

The given weights file is the weights file we intend to use for the 1% survey.

### Loading and Processing DESI Data

QuasarNP has a built in function designed to load and process single DESI exposures into the form expected by the network model.

You can load DESI data in two ways. `load_desi_daily` will load
`cframe` files that are saved in `desi/spectro/redux/daily/exposures{night}/{exp_id}`:

```python
from quasarnp.io import load_desi_daily

night, exp_id = "20210107", "00071246"
spec_number = 0
data, w = load_desi_daily(night, exp_id, spec_number)
```

`w` here defines an array of indices for which spectra are **kept** after the data reduction.
This is useful to know which spectra were removed, per the original QuasarNet code we remove any spectra where there are zero weights.


`spec_number` must range from 0-9 and defines which spectrograph to load.
The final parameter for `load_desi_daily` (not set in the above example)
is `fibers` which is defined as a numpy array of length 500, where an item is set to 1 if we should load that
fiber and 0 if we should not. Ex, this will also load the first 400 fibers
as well as the 500th fiber:
```python
f = np.zeros(500, dtype=bool)
f[0:400] = 1
f[-1] = 1
data, w = load_desi_daily(night, exp_id, spec_number, f)
```

`load_desi_daily` will handle everything involved in loading and processing the
data. It will load all three wavelength bands and then rebin them to match the 443
input bins that QuasarNet/NP expects. This rebinning is currently done using the
same code that QuasarNet does. This may change in the future, but the public
facing API will not.

If you would prefer to load `cframe` files from an arbitrary folder, `quasarnp` also provides `load_desi_exposure` which will load from an arbitrary directory.
To load the same data as above you could instead use

```python
from quasarnp.io import load_desi_exposure

night, exp_id = "20210107", "00071246"
dir_name = f"/global/cfs/cdirs/desi/spectro/redux/daily/exposures/{night}/{exp_id}"
spec_number = 0
data, w = load_desi_exposure(dir_name, spec_number)
```

Once that is done it will process the data by subtracting the ivar-weighted mean
from the data and dividing by the rms value. After this is complete the data is
ready to be run through QuasarNP.

For more details some of the specifics of the data normalization, and how it was done in QuasarNet, see [`qn_on_desi.md`](https://github.com/desihub/QuasarNP/blob/main/qn_on_desi.md)

### Running DESI Data
Once the data has ben processed into a form expected by the network we will
make a prediction. After making a prediction through the network, it is necessary
to interpret the output numbers into a human readable format. `process_preds`
exists for this reason.

The output of `process_preds` are in arrays of len `nspec` where each element
corresponds to the spectra that is in that position in the input data array.

The output `c_line` is the one we care about for classifying Quasars. This
output defines the confidence that each input line appears in the input spectra.

In the following code snippet, `c_thresh` defines the threshold above which we
are certain that QuasarNet detected the line in the spectra, and `n_thresh` defines
the number of lines that must be detected for the object to be classified a Quasar.
It is generally reasonable to set these to values of `0.5` and `1`, but you can
increase both to tighten the restrictions on what is and isn't a quasar.

`c_line_bal` provides the same data for the BAL line(s).

`z_line` provides the predicted redshift of each line, and is generally only
useful for things that are identified as quasars, i.e. objects where the line
actually appears. Outside of that the redshift predictions are nonsense since
the line didn't appear.

The first line here expands the dimensions of the `data` array to match those
that are expected by predict.

```python
from quasarnp.utils import process_preds

data = data[:, :, None]
p = model.predict(data)
c_line, z_line, zbest, c_line_bal, z_line_bal = process_preds(p, lines, lines_bal)

c_thresh = 0.5
n_thresh = 1
is_qso = np.sum(c_line > c_thresh, axis=0) >= n_thresh
```

`is_qso` then defines an array of length `nfibers` that will tell you which ones
 that are identified as being a quasar (set to 1) or not (set to 0).