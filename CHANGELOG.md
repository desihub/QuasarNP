# Changelog

This the log of changes to [QuasarNP](https://github.com/desihub/QuasarNP).

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2022-07-20
### Added
- [#b5562d69] Add `read_spall` from QuasarNET.
- [#5372189d] Tests for rebinning spectra.
- [#79cc5e48] `conv1d` can now accept two different padding modes.


### Changed
- [#acd75825] Renormalizing spectra is now its own function (`utils.renormalize`)
- [#e37dcde3] Rebin now also works on eBOSS data, not just DESI data.
- [#c4dd52c6] `load_model` and `load_file` now dynamically determine the number
of convolution layers in the trained model.
- [#7e3f7141] `load_model` and `load_file` now dynamically determine the padding mode
and stride of each convolution layer

## [0.1.3] - 2021-10-04
### Added
- Unit tests.

### Changed
- Fix PEP-8 style errors in layers.py.

## [0.1.2] - 2021-07-12
### Changed
- Fix a bug with `rows == None` sometimes not working as intended, and instead
use `rows is None`
- Fix PEP-8 style errors in io.py.


## [0.1.1] - 2021-06-04
### Added
- Add version string.
- Add Sphinx docfiles.

## [0.1.0] - 2021-05-26
Initial release.