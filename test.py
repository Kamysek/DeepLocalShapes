from deep_ls.data import determine_cubes_for_sample

import numpy as np
import time 

# start = time.time()

filename = "/home/josefkamysek/DeepLocalShapes/temp_1a6ad7a24bb89733f412783097373bdc.npz"
# before_npz = np.load(filename)

# determine_cubes_for_sample(filename=filename, box_size=1, cube_size=32, radius=1.5)

# end = time.time() - start

# print(end)

after_npz = np.load(filename, allow_pickle=True)

print("Done.")