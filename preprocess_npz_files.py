from deep_ls.data import determine_cubes_for_sample

import numpy as np
import time 
import os

counter = 0 
for file in os.listdir("/home/josefkamysek/DeepLocalShapes/data/SdfSamples/ShapeNetV2/02691156/"):
    
    if counter == 100:
        print("Done.")
        exit(0)

    file_with_path = os.path.join("/home/josefkamysek/DeepLocalShapes/data/SdfSamples/ShapeNetV2/02691156/", file)

    start_time = time.time()

    determine_cubes_for_sample(filename=file_with_path, box_size=1, cube_size=32, radius=1.5)

    end_time = time.time() - start_time
    print(end_time)

    counter += 1

print("Done.")