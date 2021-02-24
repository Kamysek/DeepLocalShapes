#!/usr/bin/env python3
# Based on: https://github.com/facebookresearch/DeepSDF using MIT LICENSE (https://github.com/facebookresearch/DeepSDF/blob/master/LICENSE)
# Copyright 2021-present Philipp Friedrich, Josef Kamysek. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import functools
import deep_ls
import deep_ls.workspace as ws

import torch.multiprocessing as mp
from scipy.spatial import cKDTree
import numpy as np


def trainer(center_point, sdf_tree, sdf_grid_radius, latent, sdf_data, sdf_grid_indices, loss_sum, loss_lock, decoder, loss_l1, l2reg):
    
    loss = 0.0

    near_sample_indices = sdf_tree.query_ball_point(x=[center_point[1]], r=sdf_grid_radius, p=np.inf) 
    
    num_sdf_samples = len(near_sample_indices[0])
    
    if num_sdf_samples < 1: 
        return
    
    code = latent[center_point[0]].cuda()

    sdf_gt = sdf_data[near_sample_indices[0], 3].unsqueeze(1)

    sdf_gt = torch.tanh(sdf_gt)

    transformed_sample = sdf_data[near_sample_indices[0], :3] - sdf_grid_indices[center_point[0]] 
    
    transformed_sample.requires_grad = False
    
    decoder.requires_grad = False
    
    code = code.expand(1, 125)
    
    code = code.repeat(transformed_sample.shape[0], 1)
    
    decoder_input = torch.cat([code, transformed_sample.cuda()], dim=1).float().cuda()

    pred_sdf = decoder(decoder_input)
    
    loss += loss_l1(pred_sdf, sdf_gt.cuda()) / len(sdf_grid_indices)

    if l2reg:
        loss += 1e-4 * torch.mean(latent.pow(2))
       
    loss.backward()

    with loss_lock:
        loss_sum.value += loss
        return


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    cube_size,
    box_size,
    voxel_radius,
    test_sdf,
    stat,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    save_intermediate=0
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    sdf_grid_indices = deep_ls.data.generate_grid_center_indices(cube_size=cube_size, box_size=box_size)
    sdf_grid_radius = voxel_radius * ((box_size * 2) / cube_size)

    if type(stat) == type(0.1):
        latent = torch.ones(len(sdf_grid_indices), latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()
        raise NotImplementedError # TODO

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()

        sdf_data = deep_ls.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        )

        xyz = sdf_data[:, 0:3]
    
        sdf_tree = cKDTree(xyz)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()
        
        if __name__ == '__main__': 
            # Shared value counter and lock
            mp.set_start_method('spawn', force=True)
            manager = mp.Manager()
            loss_sum = manager.Value('f', 0)
            loss_lock = manager.Lock()
            
            # Create Pool for multiprocessing
            start = time.time()
            pool = mp.Pool()

            # Apply map on array of center points
            res = pool.map(functools.partial(trainer,                                
                            sdf_tree = sdf_tree, 
                            sdf_grid_radius = sdf_grid_radius,
                            latent = latent, 
                            sdf_data = sdf_data, 
                            sdf_grid_indices = sdf_grid_indices, 
                            loss_sum = loss_sum, 
                            loss_lock = loss_lock, 
                            decoder = decoder, 
                            loss_l1 = loss_l1), 
                            enumerate(sdf_grid_indices))

            pool.close()
            pool.join()

            logging.info("Multiprocessing Time {}".format(time.time() - start))

        loss = loss_sum
       
        optimizer.step()

        if e % 1 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())

        loss_num = loss.cpu().data.numpy()

        if save_intermediate > 0 and e % save_intermediate == 0:
            logging.debug("Saving intermediate Reconstruction result.")
            start = time.time()
            with torch.no_grad():
                deep_ls.mesh.create_mesh(
                    decoder, latent, cube_size, box_size, mesh_filename, N=128, max_batch=int(2 ** 18)
                )
            logging.debug("Reconstruct intermediate result took: {} seconds.".format(time.time() - start))

    return loss_num, latent


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepLS decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--save_intermediate",
        dest="save_intermediate",
        default=0,
        help="Save intermediate reconstructions each n steps. Deactivate with 0.",
    )
    deep_ls.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_ls.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]
    cube_size = specs["CubeSize"]
    box_size = specs["BoxSize"]
    voxel_radius = specs["VoxelRadius"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    if torch.cuda.device_count() > 1:
        decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    if torch.cuda.device_count() > 1:
        decoder = decoder.module.cuda()
    else:
        decoder = decoder.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames = deep_ls.data.get_instance_filenames(args.data_source, split)

    random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        logging.debug("loading {}".format(npz))

        data_sdf = deep_ls.data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                cube_size,
                box_size,
                voxel_radius,
                data_sdf,
                0.01,  # [emp_mean,emp_var],
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
                save_intermediate=int(args.save_intermediate)
            )
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            #logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    deep_ls.mesh.create_mesh(
                        decoder, latent, cube_size, box_size, mesh_filename, N=128, max_batch=int(2 ** 18)
                    )
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
