import h5py
import numpy as np

from .model import QuasarNP

def load_file(filename):
    result = {}

    with h5py.File(filename, "r") as f:
        m_weights = f['model_weights']

        for k1, v1 in m_weights.items():
            if len(v1) == 0: continue
            a = v1[k1]

            data_dict = {}
            for k2, v2 in a.items():
              # This :-2 strips off the :0 at the end of weights names.
                data_dict[k2[:-2]] = v2[()]

            result[k1] = data_dict

    return result

def load_model(filename):
    db = load_file(filename)
    return QuasarNP(db)

llmin = np.log10(3600)
llmax = np.log10(10000)
dll = 1e-3

nbins = int((llmax-llmin)/dll)
wave = 10**(llmin + np.arange(nbins)*dll)
nmasked_max = len(wave)+1