import os
import random

import icon_registration as icon
import icon_registration.networks as networks
import numpy as np
import torch
import torch.nn.functional as F
from icon_registration import DownsampleRegistration
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


class ConsistentFromMatrix(icon.RegistrationModule):
    """
    wrap an inner neural network `net` that returns an N x N+1 matrix representing
    an affine transform, into a RegistrationModule that returns a function that
    transforms a tensor of coordinates.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        matrix_phi = self.net(image_A, image_B) - self.net(image_B, image_A)

        matrix_phi_BA = torch.linalg.matrix_exp(-matrix_phi)
        matrix_phi = torch.linalg.matrix_exp(matrix_phi)

        # matrix_phi = matrix_phi / 256

        # matrix_phi += torch.tensor(
        #     [
        #         [1.0, 0.0, 0.0, 0.0],
        #         [0.0, 1.0, 0.0, 0.0],
        #         [0.0, 0.0, 1.0, 0.0],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ]
        # ).to(image_A.device)

        # matrix_phi_BA = -matrix_phi

        # for _ in range(8):
        #    matrix_phi = torch.matmul(matrix_phi, matrix_phi)
        #    matrix_phi_BA = torch.matmul(matrix_phi_BA, matrix_phi_BA)

        def transform(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [
                    tensor_of_coordinates,
                    torch.ones(shape, device=tensor_of_coordinates.device),
                ],
                axis=1,
            )
            return icon.network_wrappers.multiply_matrix_vectorfield(
                matrix_phi, coordinates_homogeneous
            )[:, :-1]

        def transform1(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [
                    tensor_of_coordinates,
                    torch.ones(shape, device=tensor_of_coordinates.device),
                ],
                axis=1,
            )
            return icon.network_wrappers.multiply_matrix_vectorfield(
                matrix_phi_BA, coordinates_homogeneous
            )[:, :-1]

        return transform, transform1


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


def multiscale_model(input_shape, lmbda):

    inner_net = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
    inner_net2 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
    inner_net3 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))

    threestep_consistent_net = icon.losses.BendingEnergyNet(
        UnwrapHalfwaynet(
            TwoStepInverseConsistent(
                icon.DownsampleRegistration(
                    icon.DownsampleRegistration(inner_net, 3), 3
                ),
                TwoStepInverseConsistent(
                    icon.DownsampleRegistration(inner_net2, 3), inner_net3
                ),
            )
        ),
        icon.LNCC(sigma=5),
        lmbda=lmbda,
    )
    threestep_consistent_net.assign_identity_map(input_shape)
    return threestep_consistent_net


def all_affine_model(input_shape, lmbda):

    inner_net = ConsistentFromMatrix(networks.ConvolutionalMatrixNet(dimension=3))
    inner_net2 = ConsistentFromMatrix(networks.ConvolutionalMatrixNet(dimension=3))

    threestep_consistent_net = icon.losses.BendingEnergyNet(
        UnwrapHalfwaynet(
            TwoStepInverseConsistent(
                icon.DownsampleRegistration(inner_net, 3), inner_net2
            ),
        ),
        icon.LNCCOnlyInterpolated(sigma=5),
        lmbda=lmbda,
    )
    threestep_consistent_net.assign_identity_map(input_shape)
    return threestep_consistent_net


def multiscale_affine_model(input_shape, lmbda):

    inner_net = ConsistentFromMatrix(networks.ConvolutionalMatrixNet(dimension=3))
    inner_net2 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
    inner_net3 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))

    threestep_consistent_net = icon.losses.BendingEnergyNet(
        UnwrapHalfwaynet(
            TwoStepInverseConsistent(
                icon.DownsampleRegistration(
                    icon.DownsampleRegistration(inner_net, 3), 3
                ),
                TwoStepInverseConsistent(
                    icon.DownsampleRegistration(inner_net2, 3), inner_net3
                ),
            )
        ),
        icon.LNCC(sigma=5),
        lmbda=lmbda,
    )
    threestep_consistent_net.assign_identity_map(input_shape)
    return threestep_consistent_net


def pretrained_affine_deformable_model(input_shape, lmbda):

    affine_inner_net = ConsistentFromMatrix(
        networks.ConvolutionalMatrixNet(dimension=3)
    )
    affine_inner_net2 = ConsistentFromMatrix(
        networks.ConvolutionalMatrixNet(dimension=3)
    )

    inner_net = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
    inner_net2 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))

    threestep_consistent_net = icon.losses.BendingEnergyNet(
        UnwrapHalfwaynet(
            TwoStepInverseConsistent(
                DownsampleRegistration(affine_inner_net, 3),
                TwoStepInverseConsistent(
                    affine_inner_net2,
                    TwoStepInverseConsistent(
                        icon.DownsampleRegistration(inner_net, 3), inner_net2
                    ),
                ),
            )
        ),
        icon.LNCC(5),
        lmbda=lmbda,
    )
    threestep_consistent_net.assign_identity_map(input_shape)

    threestep_consistent_net.train()
    return threestep_consistent_net
