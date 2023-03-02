import datetime
import os
import random
import sys
import tempfile

import footsteps
import icon_registration as icon
import icon_registration.data
import icon_registration.networks as networks
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
from icon_registration.config import device
from torch.nn.parallel import DistributedDataParallel as DDP

import multiscale_constr_model

from multiscale_constr_model import (
    ConsistentFromMatrix,
    ICONSquaringVelocityField,
    TwoStepInverseConsistent,
    UnwrapHalfwaynet,
    all_affine_model
)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=1800),
    )


def cleanup():
    dist.destroy_process_group()


def make_model():
    input_shape = [1, 1, 130, 155, 130]

    lmbda = 0.0001

    threestep_consistent_net = multiscale_constr_model.all_affine_model(input_shape, lmbda)
    threestep_consistent_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        threestep_consistent_net
    )
    qq = torch.nn.Module()
    qq.module = threestep_consistent_net
    weights_path = "results/brain_affine_train/network_weights_23400"
    qq.load_state_dict(torch.load(weights_path), strict=True)

    with torch.no_grad():
        threestep_consistent_net.regis_net.net.netPsi.net.dense3.weight /= 2
        threestep_consistent_net.regis_net.net.netPsi.net.dense3.bias /= 2



    inner_net = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
    inner_net2 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))

    threestep_consistent_net = icon.losses.BendingEnergyNet(
            UnwrapHalfwaynet(
                TwoStepInverseConsistent(
                    threestep_consistent_net.regis_net.net.netPhi,
                    TwoStepInverseConsistent(
                        threestep_consistent_net.regis_net.net.netPsi,
                        TwoStepInverseConsistent(
                            icon.DownsampleRegistration(inner_net, 3),
                            inner_net2
                        )
                    )
                )
            ),
            icon.LNCC(5),
            lmbda=lmbda
    )
    threestep_consistent_net.assign_identity_map(input_shape)

    threestep_consistent_net.train()
    return threestep_consistent_net


BATCH_SIZE = 3


def make_batch(dataset, rank):
    image = torch.cat([random.choice(dataset) for _ in range(BATCH_SIZE)])
    image = image.to(rank)
    image = image / torch.max(image)
    return image


def train_batchfunction(net, optimizer, steps=100000, unwrapped_net=None, rank=None):
    """A training function intended for long running experiments, with tensorboard logging
    and model checkpoints. Use for medical registration training
    """
    dataset = torch.load(
        "/playpen-ssd/tgreer/ICON_brain_preprocessed_data/stripped/brain_train_2xdown_scaled"
    )
    if rank == 0:
        import footsteps
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            footsteps.output_dir
            + "/"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            flush_secs=30,
        )

    for iteration in range(0, steps):
        optimizer.zero_grad()
        moving_image, fixed_image = (
            make_batch(dataset, rank),
            make_batch(dataset, rank),
        )
        loss_object = net(moving_image, fixed_image)

        loss = torch.mean(loss_object.all_loss)
        loss.backward()
        if rank == 0:
            print(icon.train.to_floats(loss_object))
            icon.train.write_stats(writer, loss_object, iteration)
        optimizer.step()

        if iteration % 300 == 0 and rank == 0:
            # torch.save(
            #    optimizer.state_dict(),
            #    footsteps.output_dir + "optimizer_weights_" + str(iteration),
            # )
            torch.save(
                net.state_dict(),
                footsteps.output_dir + "network_weights_" + str(iteration),
            )


def per_process_function(rank, world_size):
    setup(rank, world_size)
    model = make_model().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.00005)

    train_batchfunction(ddp_model, optimizer, steps=50000, rank=rank)


if __name__ == "__main__":
    n_gpus = 4
    mp.spawn(per_process_function, args=(n_gpus,), nprocs=n_gpus, join=True)
