import numpy as np

def Load_file(path):
    data_type = path.split('.')[-1]
    if data_type == 'txt':
        data = np.loadtxt(path, dtype=np.float32)
    elif data_type == 'npy':
        data = np.array(np.load(path), dtype=np.float32)
    return data

