#!/usr/bin/env python3
# Based on: https://github.com/facebookresearch/DeepSDF using MIT LICENSE (https://github.com/facebookresearch/DeepSDF/blob/master/LICENSE)
# Copyright 2021-present Philipp Friedrich, Josef Kamysek. All Rights Reserved.

import functools
import json
import logging
import math
import os
import signal
import sys
import time
import warnings
import time
import deep_ls
import deep_ls.workspace as ws
import torch
import torch.multiprocessing as mp
import torch.utils.data as data_utils
from scipy.spatial import cKDTree
import numpy as np

from tqdm import tqdm

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):
    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):
    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):
    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):
    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):
    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):
    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
        experiment_directory,
        loss_log,
        lr_log,
        timing_log,
        lat_mag_log,
        param_mag_log,
        epoch,
):
    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):
    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):
    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return loss_log, lr_log, timing_log, lat_mag_log, param_mag_log


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    sum_for_mean = 0
    for i in latent_vectors:
        sum_for_mean += torch.mean(torch.norm(i.weight.data.detach(), dim=1)).item()

    return sum_for_mean / len(latent_vectors)


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def trainer(center_point, sdf_tree, sdf_grid_radius, lat_vecs, sdf_data, indices, cube_size, outer_sum, outer_lock, decoder, loss_l1, do_code_regularization, code_reg_lambda, epoch):
    inner_sum = 0.0
    
    # Get all indices of the samples that are within the L-radius around the cell center.
    near_sample_indices = sdf_tree.query_ball_point(x=[center_point[1]], r=sdf_grid_radius, p=np.inf) 
    
    # Get number of samples located within the L-radius around the cell center
    num_sdf_samples = len(near_sample_indices[0])
    if num_sdf_samples < 1: 
       return
    
    # Extract code from lat_vecs
    code = lat_vecs((center_point[0] + indices[0].cuda() * (cube_size**3)))

    # Get groundtruth sdf value
    sdf_gt = sdf_data[near_sample_indices[0], 3].unsqueeze(1)
    sdf_gt = torch.tanh(sdf_gt).cuda()
    
    transformed_sample = sdf_data[near_sample_indices[0], :3] - center_point[1]
    transformed_sample.requires_grad = False
    
    code = code.expand(1, 125)
    code = code.repeat(transformed_sample.shape[0], 1)
    
    decoder_input = torch.cat([code, transformed_sample.cuda()], dim=1).float()
    
    # Get network prediction of current sample
    pred_sdf = decoder(decoder_input) 
    
    # f_theta - s_j
    inner_sum = loss_l1(pred_sdf, sdf_gt) / num_sdf_samples

    # Right most part of formula (4) in DeepLS ->  + 1/sigma^2 L2(z_i)
    if do_code_regularization and num_sdf_samples != 0:
        l2_size_loss = torch.sum(torch.norm(code, dim=0))

        reg_loss = (code_reg_lambda * min(1.0, epoch / 100) * l2_size_loss) / num_sdf_samples

        inner_sum = inner_sum.cuda() + reg_loss.cuda()

    inner_sum.backward()

    with outer_lock:
        outer_sum.value += inner_sum.item()
        return

def main_function(experiment_directory, continue_from, batch_split):
    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + str(specs["Description"]))

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)
    
    cube_size = get_spec_with_default(specs, "CubeSize", 30)
    box_size = get_spec_with_default(specs, "BoxSize", 1)
    voxel_radius = get_spec_with_default(specs, "VoxelRadius", 1.5)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    if torch.cuda.device_count() > 1:
        decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = deep_ls.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False
    )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    # Generate grid indices of cube
    sdf_grid_indices = deep_ls.data.generate_grid_center_indices(cube_size=cube_size, box_size=box_size)

    # voxel_radius is defined as 1.5 times the voxel side length (see DeepLS sec. 4.1) since that value provides
    # a good trade of between accuracy and efficiency
    sdf_grid_radius = voxel_radius * (((box_size * 2) / cube_size) / 2)

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    # TODO check if there is something better than Embedding to store codes.
    # TODO Not sure if max_norm=code_bound is necessary
    lat_vecs = torch.nn.ModuleList()
    for scene in range(num_scenes):
        embedding = torch.nn.Embedding(cube_size**3, latent_size, max_norm=code_bound)
        torch.nn.init.normal_(
            embedding.weight.data,
            0.0,
            get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
        )
        embedding.requires_grad = True
        lat_vecs.append(embedding)

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    # TODO time loss on cuda and without
    loss_l1 = torch.nn.L1Loss(reduction="sum").cuda()

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs[0].num_embeddings * lat_vecs[0].embedding_dim * num_scenes,
            lat_vecs[0].num_embeddings * num_scenes,
            lat_vecs[0].embedding_dim * num_scenes,
        )
    )

    for epoch in range(start_epoch, num_epochs + 1):
        
        start = time.time()
        
        logging.info("epoch {}...".format(epoch))

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        current_scene = 0
        scene_avg_loss = 0.0

        for sdf_data, indices in tqdm(sdf_loader):

            # Get correct lat_vecs embedding and load to cuda
            temp_lat_vec = lat_vecs[indices]
            
            # Sdf_data contains n samples per scene
            sdf_data = sdf_data.reshape(-1, 4)            
            sdf_data.requires_grad = False

            # Amount of extracted sdf samples
            num_sdf_samples_total = sdf_data.shape[0]

            # Extract point coordinates of sdf value
            xyz = sdf_data[:,:3]
            # Build cKDTree to access the indices within a certain radious of a point in a very fast fashion
            sdf_tree = cKDTree(xyz)

            # Counter for debug purposes
            total_batches_used = 0
            samples_used = 0
            empty_grid_cells = 0

            # Calculate batch size
            batch_size = int(len(sdf_grid_indices) / batch_split)

            outer_sum = 0.0
            optimizer_all.zero_grad()

            for center_point_index in range(0, len(sdf_grid_indices), batch_size):

                inner_sum = 0.0
                inputs = []
                sdf_gts = []
                batches_used = 0

                for batch in range(batch_size):
                    # Calculate index in temporary embedding 
                    index = center_point_index + batch

                    # Get all indices of the samples that are within the L-radius around the cell center.
                    near_sample_indices = sdf_tree.query_ball_point(x=[sdf_grid_indices[index]], r=sdf_grid_radius, p=np.inf)
                    
                    # Get number of samples located samples within the L-radius around the cell center
                    near_sample_indices = near_sample_indices[0]
                    if len(near_sample_indices) < 1: 
                        continue
                    
                    samples_used += len(near_sample_indices)
                    
                    sdf_gt = sdf_data[near_sample_indices, 3].unsqueeze(1)
                    
                    sdf_gts.append(torch.tanh(sdf_gt))

                    transformed_sample = sdf_data[near_sample_indices, :3] - sdf_grid_indices[index] 
                    transformed_sample.requires_grad = False
                    
                    code = temp_lat_vec((torch.empty(1).fill_(index)).long())
                    code = code.expand(1, 125)
                    code = code.repeat(transformed_sample.shape[0], 1)
                    
                    inputs.append(torch.cat([code, transformed_sample], dim=1).float().cuda())
                    
                    batches_used += 1
                
                # Get network prediction of current sample
                if len(inputs) < 1:
                    empty_grid_cells += 1
                    continue

                total_batches_used += batches_used

                decoder_input = torch.cat(inputs, dim=0)

                pred_sdf = decoder(decoder_input) 

                # f_theta - s_j
                sdf_gt = torch.cat(sdf_gts, dim=0)
                inner_sum = loss_l1(pred_sdf.squeeze(0), sdf_gt.cuda()) / decoder_input.shape[0]

                # Right most part of formula (4) in DeepLS ->  + 1/sigma^2 L2(z_i)
                if do_code_regularization and num_sdf_samples_total != 0:
                    l2_size_loss = torch.sum(torch.norm(code, dim=0))

                    reg_loss = (code_reg_lambda * min(1.0, epoch / 100) * l2_size_loss) / decoder_input.shape[0]

                    inner_sum = inner_sum + reg_loss

                inner_sum.backward()

                outer_sum += inner_sum.item()

            scene_avg_loss += outer_sum

            current_scene += 1

            #logging.info("Scene {}, Scence Index {}, loss = {}".format(current_scene, indices.item(), outer_sum))
            #logging.info("Total batches used {} total samples {}".format(total_batches_used, samples_used))
            #logging.info("Empty grid cells {}".format(empty_grid_cells))

            loss_log.append(outer_sum)

            optimizer_all.step()

        logging.info("Epoch took {} seconds".format(time.time() - start))            
        logging.info("Epoch scene average loss: {}".format((scene_avg_loss / current_scene)))
        
        end = time.time()
        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        # TODO check what other functions do with lat_vecs and adapt if needed.
        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:
            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )
        """
        # FOR DEBUGGING ONLY!
        logging.debug("Trying to reconstruct with trained model")
        with torch.no_grad():
            debug_file_name = "/home/philippgfriedrich/DeepLocalShapes/examples/sofas/Reconstructions/2000/Meshes/ShapeNetV2/04256520/trained_1037fd31d12178d396f164a988ef37cc"
            lat_vec_mesh = np.array(lat_vecs.cpu().weight.data)
            deep_ls.mesh.create_mesh(
                decoder, lat_vec_mesh, cube_size, box_size, debug_file_name, N=128, max_batch=int(2 ** 18)
            )
            logging.debug("total time: {}".format(time.time() - start))"""
        

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepLS autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
             + "from the latest running snapshot, or an integer corresponding to "
             + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
             + "processed separately, with gradients accumulated across all "
             + "subbatches. This allows for training with large effective batch "
             + "sizes in memory constrained environments.",
    )

    deep_ls.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_ls.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
