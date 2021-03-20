#!/usr/bin/env python3
# Based on: https://github.com/facebookresearch/DeepSDF using MIT LICENSE (https://github.com/facebookresearch/DeepSDF/blob/master/LICENSE)
# Copyright 2021-present Philipp Friedrich, Josef Kamysek. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import deep_ls.workspace as ws

from collections import defaultdict
from tqdm import tqdm

def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                        os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[0][:, 3])
    return tensor[~tensor_nan, :]

# def remove_tobig_tosmall(tensor):
#     xyz = tensor[0][:, 0:3]
#     tensor_tobig = torch.where(xyz > 1)[0].numpy()
#     tensor_tosmall = torch.where(xyz < -1)[0].numpy()
#     tensor_tobig_tosmall = np.concatenate((tensor_tobig, tensor_tosmall))

#     tensor = np.delete(tensor[0].numpy(), tensor_tobig_tosmall, axis=0)
#     tensor = torch.from_numpy(tensor).double()
#     return tensor 


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"].double())
    neg_tensor = torch.from_numpy(npz["neg"].double())

    return [pos_tensor, neg_tensor]


def create_dict_and_index(samples):
    grid_samples = defaultdict(list)
    for indx, sample in enumerate(samples):
        for grid_index in sample[-1]:
            grid_samples[grid_index].append(indx)
        
    return grid_samples

   

def add_cubes_to_samples(samples, voxel_size, grid_indices, radius):
    
    temp_samples = np.zeros((samples.shape[0],31))

    for index, element in enumerate(tqdm(samples)):
        # Extract xyz coordinates of current sample
        x, y, z = element[0:3]

        cube_indices = np.zeros((27))
        # Determine x,y,z entry in matrix
        x_entry = int(np.floor((x+1) / voxel_size)) 
        y_entry = int(np.floor((y+1) / voxel_size))
        z_entry = int(np.floor((z+1)/ voxel_size))

        # Determine subcube index and get center point of subcube index
        N = 32
        counter = 0 
        for x_change in [-1, 0, 1]:
            for y_change in [-1, 0, 1]:
                for z_change in [-1, 0, 1]:
                    tmp_x = x_entry + x_change
                    tmp_y = y_entry + y_change
                    tmp_z = z_entry + z_change
                    
                    if min(tmp_x, tmp_y, tmp_z) >= 0 and max(tmp_x, tmp_y, tmp_z) < 32:
                        tmp_grid_index = tmp_z + N * (tmp_y + N * tmp_x)
                        tmp_grid_xyz  = grid_indices[tmp_grid_index]
                        if abs(tmp_grid_xyz[0]-x) < radius and abs(tmp_grid_xyz[1] - y) < radius and abs(tmp_grid_xyz[2] - z):
                            cube_indices[counter] = tmp_grid_index + 1
                    counter += 1
        temp_samples[index] = np.concatenate((element, cube_indices))
        
    return temp_samples


def determine_cubes_for_sample(filename, box_size, cube_size, radius=1.5):
    # Determine voxel_size
    voxel_size = 2 * box_size / cube_size    
    
    # Load npz file
    npz = np.load(filename)
    pos = npz["pos"]
    neg = npz["neg"] 

    grid_indices = generate_grid_center_indices(cube_size=32, box_size=1)
    # Modify npz file
    radius = radius * voxel_size

    print("Processing positives samples...")
    pos = add_cubes_to_samples(pos, voxel_size, grid_indices, radius)
    print("Processing negative samples...")
    neg = add_cubes_to_samples(neg, voxel_size, grid_indices, radius)
                            
    # Store samples  back
    filename = filename[:-4] + "_temp.npz"
    np.savez(filename, pos=pos, neg=neg)
            

def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename, allow_pickle=True)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]).double())
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]).double())

    # pos_tensor = remove_tobig_tosmall(pos_tensor)
    # neg_tensor = remove_tobig_tosmall(neg_tensor)

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half, dtype=torch.double) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half, dtype=torch.double) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    # dict = new_func(samples, radius)

    return samples


def generate_grid_center_indices(cube_size=32, box_size=1):
    # Divide space into equally spaced subspaces and calculate center position of subspace
    voxel_centers = np.linspace(-box_size, box_size, cube_size, endpoint=False, dtype=np.float)
    voxel_centers += box_size / cube_size

    # Create grid indices
    return np.vstack(np.meshgrid(voxel_centers, voxel_centers, voxel_centers)).reshape(3, -1).T


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind: (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half, dtype=torch.double) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind: (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            split,
            subsample,
            load_ram=False,
            print_filename=False,
            num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]).double())
                # pos_tensor = remove_tobig_tosmall(pos_tensor)
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]).double())
                # neg_tensor = remove_tobig_tosmall(neg_tensor)
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx
