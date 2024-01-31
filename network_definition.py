import os
import random

import footsteps
import icon_registration as icon
import icon_registration.networks as networks
import torch

import multiscale_constr_model


BATCH_SIZE = 2
GPUS = 4


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
