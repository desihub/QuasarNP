# Processing DESI Data with QuasarNet

This markdown file aims to track the process by which QuasarNet loads and then runs
DESI data. QuasarNet by default loads the coadded exposures, but since we hope to use
QuasarNet to determine if we should reobserve a QSO we would rather run it on the
raw cframe files instead.
In order to do this it is necessary to understand how QuasarNet uses the input DESI
files, and what (if any) preprocessing is done to the data.

## Theory

In the QuasarNet paper ([Busca et al. 2018](https://arxiv.org/abs/1808.09955)) the outlined process for running
arbitrary spectroscopic data involves the following steps:

1. Resampling the data to a 443 pixel size grid, equally spaced in log-wavelength between 360nm and 1 micrometer
2. Renormalize spectra by subtracting from the flux the weighted mean, then dividing by the weighted rms.
    1.  Inverse variance used for weighting.

And that's it! Note, however, that eBOSS data was on a larger grid
(3600-10000 angstroms) whereas DESI is 3600-9800 angstroms.
Let's take a look at the code to see how this resampling and renormalization
is done.

 ## [James Farr's DESI Code](https://github.com/jfarr03/QuasarNET)

 We can determine in what order and how this is done for DESI
 by examining the code.

 ### Preproccessing the data
 The first step to running DESI code through QN is to process the data with
 [`parse_data_desisim`](https://github.com/jfarr03/QuasarNET/blob/master/bin/parse_data_desisim)

 `parse_data_desisim` respectively calls `io.read_desi_spectra_list` which then
 in turn calls `read_desi_spectra`
 before doing some other things (which we will return to shortly)

 ### [`io.read_desi_spectra`]((https://github.com/jfarr03/QuasarNET/blob/master/py/quasarnet/io.py#L505))
 We first create a boolean array for indexing that will determine which spectra
 we keep and which we discard from the loaded file.
If we pass a target bit (`tb`) then we set all indices where the QSO target bit
is set to 1, which keeps all spectra that are quasar targets.

After this we reach

```python
nspec_init = wqso.sum()
if nspec_init == 0: return None
```
which will return immediately if we did not pass a `tb` since `wqso` is initialized
to an array of zeroes.
This is an implicit (soft) assertion that we *must* pass a `tb`.

Assuming we pass the above checkpoint we load the `TARGETID` and `SPID0`, `SPID1`
and `SPID2` but only for where `wqso` is set to 1. If spectra have duplicated metadata they get removed.

Now we get to the part where the actual spectra is loaded. Each of the three
bands is loaded separately. For each band the spectra is rebinned
first to the new wavelength grid.

Rebinning is done using `utils.rebin_wave`.
This is a little bit of a misnomer, since `rebin_wave` does not actually do the
rebinning itself, it rather takes the old wavelength grid and returns the
new wavelength grid upon which to evaluate the data. The `wave_grid_in` that is
passed to `rebin_wave` is given in the FITS file as
`h["{}_WAVELENGTH".format(band)]`

### [`utils.rebin_wave`](https://github.com/jfarr03/QuasarNET/blob/b2f62a9b7be6511cf83770f00598984a784f00de/py/quasarnet/utils.py#L178)

This is the entire contents of `rebin_wave`:

```python
bins = np.floor((np.log10(wave_grid_in)-wave_out.llmin)/wave_out.dll).astype(int)
w = (bins>=0) & (bins<wave_out.nbins)

return bins, w
```

nbins is defined as

```python
nbins = int((llmax-llmin)/dll)
```

where when we run this on desi spectra, `wave_out.llmin = np.log10(3600.)`,
`wave_out.llmax = np.log10(9800.)` and `wave_out.dll = 1.e-3`.

Here's how this works (to the best of my knowledge):
1. ``(np.log10(wave_grid_in)-wave_out.llmin)/wave_out.dll)`` finds the new x-space
position of each grid_in position, starting at `llmin`.
2. `np.floor(*)` then reduces this to simply a bin number in the new grid space.
3. `w = (bins>=0) & (bins<wave_out.nbins)` is simply an indexing array that
only sets to 1 all values that within the out_grid nbins.

`bins` and `w` are therefore of length `number of bins in h_wave` and define which
bin each observation actually falls into and whether that observation is within
the bin range defined by the wave grid.

Note that since DESI only goes up to 9800 Angstroms, this only produces 434
wavelength bins, rather than 443.

I notice here a potential problem where this could possibly throw away data outside
the upper and lower bounds, but that in theory shouldn't happen. Although if it
does not happen, is `w` strictly necessary?

Once the data has been assigned to bins, we do some data processing before the
actual rebinnng. The flux and ivar are loaded for each band. Values of flux
that are `NaN` have their ivar and flux set to zero.

The flux is rescaled by multiplying each flux by its ivar. The data is then
rebinned using

```python
for i,t in enumerate(tids):
    c = np.bincount(bins, weights = ivfl_aux[i,:])
    fl[i,:len(c)] += c
    c = np.bincount(bins, weights = iv_aux[i,:])
    iv[i,:len(c)] += c
```

`np.bincount` will count the occurrences of each bin number in bins, and then
add to that bin in `c` the value of `ivfl_aux` (the ivar times the flux) in that
location in the weights.

Once `fl` and `iv` are filled the flux is renormalized by dividing by the
summed ivar, that is `fl /= iv`, ignoring anywhere the ivar is 0. This is returned.

That completes `read_desi_spectra`. `read_desi_spectra_list` will remove anay
spectra that are in multiple files before then returning all of the loaded
spectra.

The last thing that `parse_data_desisim` does is to filter the `TARGETID` by
those in the `sdrq` if one is passed when running the script. Then it saves
all the spectra to a file using the spectrum id fields, that is:

```python
spid_fields['SPID0'] = 'TILEID'
spid_fields['SPID1'] = 'NIGHT'
spid_fields['SPID2'] = 'FIBER'
```

**NOTE:** Spectra are saved with the ivar and the flux in the same HDU:
```python
fliv = np.hstack((fl,iv))
```
So the first half of the data is the flux and the second is ivar, this is relevant
for loading the data.

In order to then run that data through the network you must call a different
script: [`qn_export`](https://github.com/jfarr03/QuasarNET/blob/master/bin/qn_export)

`qn_export` will load the data saved by `parse_data_desisim` using the standard
`read_data`
function.

### [`io.read_data`](https://github.com/jfarr03/QuasarNET/blob/b2f62a9b7be6511cf83770f00598984a784f00de/py/quasarnet/io.py#L874)
`read_data` will loop over the input list of files, loading each. It will remove
anything with a thingid of -1. Once all files are loaded we get the number of
cells.

```python
ncells = X.shape[1]/2.
assert ncells==round(ncells)
ncells = round(ncells)
```

Anything with zero weights is discarded (this is done by checking `X[:, ncells:]`,
which if you'll recall from the note above is the ivar for the flux in the first
half of the dataset.)

The data is then normalized as stated in the paper. The weighted average
of the flux is taken (using the iars as weights) and subtracted from the
data. The RMS of this is then taken. The data is then normalized by subtracting
the weighted mean and dividing by the RMS. Here is how it is done:

```python
mdata = np.average(X[:,:ncells], weights = X[:,ncells:], axis=1)
sdata = np.average((X[:,:ncells]-mdata[:,None])**2,
        weights = X[:,ncells:], axis=1)
sdata=np.sqrt(sdata)

[...]

X = X[:,:ncells]-mdata[:,None]
X /= sdata[:,None]
```

In between any data that has no RMS (`sdata`) flux is removed. If no truth
array is passed in we return here immediately, since the data itself is loaded.

Any code after this simply reformats the truth array and the redshift arrays into
forms expected by QuasarNet for purity/completeness or training purposes.

From there the data is in a form that can be run through the network, which is
done automatically in `qn_export`.