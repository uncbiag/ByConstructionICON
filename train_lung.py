import os
import random

import footsteps
import numpy as np

import icon_registration as icon
import icon_registration.networks as networks
import torch
import torch.nn.functional as F
import train_knee

input_shape = [1, 1, 175, 175, 175]

BATCH_SIZE = 2
GPUS = 3
ITERATIONS_PER_STEP = 50000
# ITERATIONS_PER_STEP = 30
WITH_AUGMENT = True


class lung_dataloader:
    def __init__(self, data_path, scale, batch_size, with_augment=False) -> None:
        self.current_ite = 0
        self.batch_size = batch_size
        self.with_augment = with_augment

        img = torch.load(f"{data_path}/lungs_train_{scale}_scaled", map_location="cpu")
        mask = torch.load(
            f"{data_path}/lungs_seg_train_{scale}_scaled", map_location="cpu"
        )

        self.data = torch.stack(
            [(torch.cat(d, 0) + 1) * torch.cat(m, 0) for d, m in zip(img, mask)], dim=0
        )
        self.current_idx_list = np.arange(self.data.shape[0])
        self._shuffle()

    def _shuffle(self) -> None:
        np.random.shuffle(self.current_idx_list)

    def make_batch(self):
        if self.current_ite + self.batch_size > len(self.data):
            self._shuffle()
            self.current_ite = 0

        batch = self.data[
            self.current_idx_list[self.current_ite : self.current_ite + self.batch_size]
        ]
        for i in range(self.batch_size):
            if random.random() > 0.5:
                temp = batch[i, 0]
                batch[i, 0] = batch[i, 1]
                batch[i, 1] = temp
        self.current_ite += self.batch_size

        if self.with_augment:
            return augment(batch[:, 0].cuda(), batch[:, 1].cuda())
        else:
            return batch[:, 0].cuda(), batch[:, 1].cuda()


def augment(image_A, image_B):
    identity_list = []
    for i in range(image_A.shape[0]):
        identity = torch.Tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
        idxs = set((0, 1, 2))
        for j in range(3):
            k = random.choice(list(idxs))
            idxs.remove(k)
            identity[0, j, k] = 1
        identity = identity * (torch.randint_like(identity, 0, 2) * 2 - 1)
        identity_list.append(identity)

    identity = torch.cat(identity_list)

    noise = torch.randn((image_A.shape[0], 3, 4))

    forward = identity + 0.05 * noise

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward.cuda(), grid_shape)

    warped_A = F.grid_sample(image_A, forward_grid, padding_mode="border")

    noise = torch.randn((image_A.shape[0], 3, 4))
    forward = identity + 0.05 * noise

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward.cuda(), grid_shape)
    warped_B = F.grid_sample(image_B, forward_grid, padding_mode="border")

    return warped_A, warped_B


if __name__ == "__main__":
    footsteps.initialize()

    dataloader = lung_dataloader(
        "/playpen-ssd/tgreer/ICON_lung/results/half_res_preprocessed_transposed_SI",
        scale="2xdown",
        batch_size=GPUS * BATCH_SIZE,
        with_augment=WITH_AUGMENT,
    )

    loss = train_knee.make_net(input_shape=input_shape, lmbda=0.5)

    net_par = torch.nn.DataParallel(loss).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()
    icon.train_batchfunction(net_par, optimizer, dataloader.make_batch, unwrapped_net=loss)
