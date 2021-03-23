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
from scipy.spatial import cKDTree


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
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def preprocess_samples(samples, grid_indices, radius=0.0625):
    temp_sdf_gts = []
    temp_inputs = []
    temp_input_grid_indices = []
    batch_size = 1
    xyz = samples[:,:3]
    sdf_tree = cKDTree(xyz)
    max_size = 300000
    with torch.no_grad():
        for center_point_index in range(0, len(grid_indices), batch_size):
            index = center_point_index

            # Get all indices of the samples that are within the L-radius around the cell center.
            near_sample_indices = sdf_tree.query_ball_point(x=[grid_indices[index]], r=radius, p=np.inf, return_sorted=True)
            # Get number of samples located samples within the L-radius around the cell center
            near_sample_indices = near_sample_indices[0]
            #near_sample_indices_single = near_sample_indices_single[0]
            if len(near_sample_indices) < 1: 
                continue
            sdf_gt = samples[near_sample_indices, 3]
            
            temp_sdf_gts.extend(torch.tanh(sdf_gt))

            transformed_samples = samples[near_sample_indices, :3] - grid_indices[index] 
            #transformed_samples = transformed_samples.expand((near_sample_indices, 3))
            temp_inputs.extend(transformed_samples)
            center_index = np.repeat(center_point_index, len(near_sample_indices))
            temp_input_grid_indices.extend(torch.from_numpy(center_index))
        del sdf_tree
        padding = max_size - len(temp_input_grid_indices)
        temp_input_grid_indices = torch.stack(temp_input_grid_indices)
        zeros_input_grid_indices = torch.zeros_like(temp_input_grid_indices[0])
        temp_input_grid_indices = torch.cat([temp_input_grid_indices, zeros_input_grid_indices.repeat(padding)])
        temp_inputs = torch.stack(temp_inputs) 
        zeros_inputs = torch.zeros_like(temp_inputs[0, :]).expand(1, 3)
        temp_inputs = torch.cat([temp_inputs, zeros_inputs.repeat(padding, 1)])
        temp_sdf_gts = torch.stack(temp_sdf_gts)
        zeros_gt = torch.zeros_like(temp_sdf_gts[0])
        temp_sdf_gts = torch.cat([temp_sdf_gts, zeros_gt.repeat(padding)])
        return temp_input_grid_indices, temp_inputs, temp_sdf_gts, padding


def unpack_sdf_samples(filename, subsample=None, grid_indices=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)
    grid_indices = generate_grid_center_indices(cube_size=32, box_size=1)

    input_grid_indices, inputs, groundtruths, padding = preprocess_samples(samples, grid_indices)

    return samples, input_grid_indices, inputs, groundtruths, padding


def generate_grid_center_indices(cube_size=50, box_size=2):
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
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
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
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
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
