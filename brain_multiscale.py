import os
import random

import footsteps
footsteps.initialize()
import icon_registration as icon
import icon_registration.data
import icon_registration.networks as networks
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils
from icon_registration.config import device


class ICONSquaringVelocityField(icon.RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.n_steps = 7

    def forward(self, image_A, image_B):
        velocityfield_delta_a = (
            self.net(image_A, image_B) - self.net(image_B, image_A)
        ) / 2**self.n_steps
        velocityfield_delta = velocityfield_delta_a

        for _ in range(self.n_steps):
            velocityfield_delta = velocityfield_delta + self.as_function(
                velocityfield_delta
            )(velocityfield_delta + self.identity_map)

        def transform_AB(coordinate_tensor):
            coordinate_tensor = coordinate_tensor + self.as_function(
                velocityfield_delta
            )(coordinate_tensor)
            return coordinate_tensor

        velocityfield_delta2 = -velocityfield_delta_a

        for _ in range(self.n_steps):
            velocityfield_delta2 = velocityfield_delta2 + self.as_function(
                velocityfield_delta2
            )(velocityfield_delta2 + self.identity_map)

        def transform_BA(coordinate_tensor):
            coordinate_tensor = coordinate_tensor + self.as_function(
                velocityfield_delta2
            )(coordinate_tensor)
            return coordinate_tensor

        return transform_AB, transform_BA


class TwoStepInverseConsistent(icon.RegistrationModule):
    def __init__(self, phi, psi):
        super().__init__()
        self.netPhi = phi
        self.netPsi = psi

    def forward(self, image_A, image_B):
        root_phi_AB, root_phi_BA = self.netPhi(image_A, image_B)

        A_tilde = self.as_function(image_A)(root_phi_AB(self.identity_map))
        B_tilde = self.as_function(image_B)(root_phi_BA(self.identity_map))

        psi_AB, psi_BA = self.netPsi(A_tilde, B_tilde)

        return (
            lambda coord: root_phi_AB(psi_AB(root_phi_AB(coord))),
            lambda coord: root_phi_BA(psi_BA(root_phi_BA(coord))),
        )


class UnwrapHalfwaynet(icon.RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        AB, BA = self.net(image_A, image_B)
        return AB


input_shape = [1, 1, 130, 155, 130]

lmbda = 0.0001

inner_net = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
inner_net2 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
inner_net3 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))

threestep_consistent_net = icon.losses.BendingEnergyNet(
    UnwrapHalfwaynet(
        TwoStepInverseConsistent(
            icon.DownsampleRegistration(icon.DownsampleRegistration(inner_net, 3), 3),
            TwoStepInverseConsistent(
                icon.DownsampleRegistration(inner_net2, 3), inner_net3
            ),
        )
    ),
    icon.LNCC(sigma=5),
    lmbda=lmbda,
)
threestep_consistent_net.assign_identity_map(input_shape)


net_par = torch.nn.DataParallel(threestep_consistent_net).cuda()
optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

net_par.train()

BATCH_SIZE = 5
GPUS = 3


def make_batch(dataset):
    image = torch.cat([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image


dataset = torch.load(
    "/playpen-ssd/tgreer/ICON_brain_preprocessed_data/stripped/brain_train_2xdown_scaled"
)
batch_function = lambda : (make_batch(dataset), make_batch(dataset))


icon.train_batchfunction(
    net_par,
    optimizer,
    batch_function,
    unwrapped_net=threestep_consistent_net,
    steps=50000,
)
