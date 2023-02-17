import os
import random

import icon_registration
import icon_registration as icon
import icon_registration.losses
import icon_registration.networks as networks
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from icon_registration import DownsampleRegistration
from icon_registration.config import device


def find_velocity_fields(phi):
    if hasattr(phi, "velocity_field"):
        yield phi.velocity_field
    for cell in phi.__closure__:
        if hasattr(cell.cell_contents, "__closure__"):
            for elem in find_velocity_fields(cell.cell_contents):
                yield elem


class VelocityFieldBendingEnergyNet(icon_registration.losses.BendingEnergyNet):
    def compute_bending_energy_loss(self, phi_AB_vectorfield):
        fields = list(find_velocity_fields(self.phi_AB))

        return sum(
            icon.losses.BendingEnergyNet.compute_bending_energy_loss(self, field)
            for field in fields
        )


class VelocityFieldDiffusion(icon_registration.losses.DiffusionRegularizedNet):
    def compute_bending_energy_loss(self, phi_AB_vectorfield):
        fields = list(find_velocity_fields(self.phi_AB))

        return sum(
            icon.losses.DiffusionRegularizedNet.compute_bending_energy_loss(
                self, field + self.identity_map
            )
            for field in fields
        )


class ICONSquaringVelocityField(icon.RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.n_steps = 7

    def forward(self, image_A, image_B):
        velocity_field = self.net(image_A, image_B) - self.net(image_B, image_A)
        velocityfield_delta_a = velocity_field / 2**self.n_steps
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

        transform_AB.velocity_field = velocity_field

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

        transform_BA.velocity_field = -velocity_field

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


class ExponentialMatrix(icon.RegistrationModule):
    """
    wrap an inner neural network `net` that returns an N x N+1 matrix representing
    an affine transform, into a RegistrationModule that returns a function that
    transforms a tensor of coordinates.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        matrix_phi = self.net(image_A, image_B) - torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ).to(image_A.device)

        matrix_phi_BA = torch.linalg.matrix_exp(-matrix_phi)
        matrix_phi = torch.linalg.matrix_exp(matrix_phi)

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


class FirstTransform(icon.RegistrationModule):
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
        FirstTransform(
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
        FirstTransform(
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
        FirstTransform(
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
        FirstTransform(
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


def evaluate(
    net, prefix, epochs, lr=0.001, ds1=None, ds2=None, doshow=False, fileext=".png"
):
    import footsteps

    if ds2 is None:
        ds2 = ds1
    net.train()
    net.to(device)

    optim = torch.optim.Adam(net.parameters(), lr=lr)
    curves = icon.train_datasets(net, optim, ds1, ds2, epochs=epochs)
    for i, name in enumerate(curves[0]._fields):

        plt.subplot(2, 3, i + 1)
        plt.plot([getattr(c, name) for c in curves])
        if name == "inverse_consistency_loss":
            name = "bending_energy_loss"
        plt.title(name)
    plt.tight_layout()
    if doshow:
        plt.show()
    else:
        plt.savefig(footsteps.output_dir + prefix + "curves" + fileext)
        plt.clf()

    def show(tensor):
        plt.imshow(torchvision.utils.make_grid(tensor[:12], nrow=4)[0].cpu().detach())
        plt.xticks([])
        plt.yticks([])

    image_A = next(iter(ds1))[0].to(device)[:12]
    image_B = next(iter(ds2))[0].to(device)[:12]
    with torch.no_grad():
        print(net(image_A, image_B))
        try:
            net.prepare_for_viz(image_A, image_B)
        except:
            pass
    plt.subplot(2, 2, 1)
    plt.title("Moving Images")
    show(image_A)
    plt.subplot(2, 2, 2)
    plt.title("Fixed Images")
    show(image_B)
    plt.subplot(2, 2, 3)
    plt.title("Warped Images")
    show(net.warped_image_A)
    plt.contour(
        torchvision.utils.make_grid(net.phi_AB_vectorfield[:12], nrow=4)[0]
        .cpu()
        .detach()
    )
    plt.contour(
        torchvision.utils.make_grid(net.phi_AB_vectorfield[:12], nrow=4)[1]
        .cpu()
        .detach()
    )
    plt.subplot(2, 2, 4)
    plt.title("Difference Images")
    show(net.warped_image_A - image_B)
    plt.tight_layout()
    if doshow:
        plt.show()
    else:
        plt.savefig(footsteps.output_dir + prefix + "images" + fileext)
        plt.clf()
    plt.title("Detail of deformations")

    show(net.warped_image_A * 0)
    plt.contour(
        torchvision.utils.make_grid(net.phi_AB_vectorfield[:12], nrow=4)[0]
        .cpu()
        .detach(),
        levels=np.linspace(0, 1, 35),
    )
    plt.contour(
        torchvision.utils.make_grid(net.phi_AB_vectorfield[:12], nrow=4)[1]
        .cpu()
        .detach(),
        levels=np.linspace(0, 1, 35),
    )
    if doshow:
        plt.show()
    else:
        plt.savefig(footsteps.output_dir + prefix + "deformations" + fileext)
        plt.clf()
    plt.title("Composition of A->B and B->A transforms")
    plt.contour(
        (net.phi_AB(net.phi_BA(net.identity_map)))[0, 0].cpu().detach(),
        levels=35,
    )
    plt.contour(
        (net.phi_AB(net.phi_BA(net.identity_map)))[0, 1].cpu().detach(),
        levels=35,
    )
    if doshow:
        plt.show()
    else:
        plt.savefig(footsteps.output_dir + prefix + "composition" + fileext)
        plt.clf()
    return curves
