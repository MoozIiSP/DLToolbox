import random
import numpy as np
from h5py import File

scale_range = [96, 128, 256, 512]

f = File('test.h5', 'w')


data = []
for i in range(10):
    size = random.choice(scale_range)
    # data.append(np.random.randint(0, 255, (size, size), dtype=np.uint8))
    dset = f.create_dataset(name=f'Random Data #{i}', dtype='i8', data=np.random.randint(0, 255, (size, size), dtype=np.uint8), chunks=True)
    # dset[i] = np.random.randint(0, 255, (size, size), dtype=np.uint8)


