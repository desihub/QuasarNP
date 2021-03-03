import h5py

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
