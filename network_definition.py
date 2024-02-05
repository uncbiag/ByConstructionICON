import os
import random

import footsteps
import icon_registration as icon
import icon_registration.networks as networks
import torch
import torch.nn.functional as F

import multiscale_constr_model


BATCH_SIZE = 2
GPUS = 1

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


def make_batch(dataset):
    image = torch.cat([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image


def make_net(input_shape = [1, 1, 80, 192, 192], lmbda=5):

    net = multiscale_constr_model.FirstTransform(
        multiscale_constr_model.TwoStepInverseConsistent(
            multiscale_constr_model.ConsistentFromMatrix(
                networks.ConvolutionalMatrixNet(dimension=3)
            ),
            multiscale_constr_model.TwoStepInverseConsistent(
                multiscale_constr_model.ConsistentFromMatrix(
                    networks.ConvolutionalMatrixNet(dimension=3)
                ),
                multiscale_constr_model.TwoStepInverseConsistent(
                    multiscale_constr_model.ICONSquaringVelocityField(
                        networks.tallUNet2(dimension=3)
                    ),
                    multiscale_constr_model.ICONSquaringVelocityField(
                        networks.tallUNet2(dimension=3)
                    ),
                ),
            ),
        )
    )

    loss = multiscale_constr_model.VelocityFieldDiffusion(net, icon.LNCC(5), lmbda)
    loss.assign_identity_map(input_shape)
    return loss


if __name__ == "__main__":
    footsteps.initialize()

    dataset = torch.load("/playpen/tgreer/knees_big_2xdownsample_train_set")

    batch_function = lambda: (make_batch(dataset), make_batch(dataset))

    loss = make_net()

    net_par = torch.nn.DataParallel(loss).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()
    icon.train_batchfunction(net_par, optimizer, batch_function, unwrapped_net=loss)
