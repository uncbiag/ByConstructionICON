import argparse
import os
import random
import unittest

import footsteps
import matplotlib.pyplot as plt
import numpy as np

import icon_registration as icon
import icon_registration.data
import icon_registration.itk_wrapper
import icon_registration.networks as networks
import icon_registration.pretrained_models
import icon_registration.pretrained_models.lung_ct
import icon_registration.test_utils
import itk
import torch
import torch.nn.functional as F
import torchvision.utils
from icon_registration.config import device

footsteps.initialize(output_root="evaluation_results/")
import utils


class ICONSquaringVelocityField(icon.RegistrationModule):
    def __init__(self, net, power=1):
        super().__init__()
        self.net = net
        self.n_steps = 7
        self.power = power

    def forward(self, image_A, image_B):
        velocityfield_delta_a = (
            (self.net(image_A, image_B) - self.net(image_B, image_A))
            / 2**self.n_steps
            * self.power
        )
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
        # image_A \circ root_phi_AB \circ root_phi_AB ~ image_B
        # phi_AB = root_phi_AB \circ root_phi_AB

        # image_A \circ root_phi_AB ~ image_B \circ root_phi_BA

        A_tilde = self.as_function(image_A)(root_phi_AB(self.identity_map))
        B_tilde = self.as_function(image_B)(root_phi_BA(self.identity_map))

        psi_AB, psi_BA = self.netPsi(A_tilde, B_tilde)

        # A_tilde \circ root_psi_AB ~ B_tilde \circ root_psi_BA

        # expand A_tilde

        # image_A \circ root_phi_AB \circ root_psi_AB ~ image_B \circ root_phi_BA \circ root_psi_BA

        # right compose with root_psi_AB

        # image_A \circ root_phi_AB \circ root_psi_AB \circ root_psi_AB ~ image_B \circ root_phi_BA \circ root_psi_BA \circ root_psi_AB

        # \circ root_psi_AB cancels \circ root_psi_BA on the right side

        # image_A \circ root_phi_AB \circ root_psi_AB \circ root_psi_AB ~ image_B \circ root_phi_BA

        # right compose with root_psi_AB, cancels on the right side again

        # image_A \circ root_phi_AB \circ root_psi_AB \circ root_psi_AB \circ root_phi_AB ~ image_B

        #               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #               here is our final AB transform

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


input_shape = [1, 1, 175, 175, 175]

lmbda = 0#0.001

inner_net = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
inner_net2 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
inner_net3 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3), power=0.5)
inner_net4 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))

threestep_consistent_net = icon.losses.BendingEnergyNet(
    UnwrapHalfwaynet(
        TwoStepInverseConsistent(
            inner_net,
            TwoStepInverseConsistent(
                inner_net2, TwoStepInverseConsistent(inner_net3, inner_net4)
            ),
        )
    ),
    icon.LNCC(sigma=5),
    lmbda=lmbda,
)
threestep_consistent_net.assign_identity_map(input_shape)

image_root = "/playpen-raid1/lin.tian/data/lung/dirlab_highres_350"
landmark_root = "/playpen-raid1/lin.tian/data/lung/reg_lung_2d_3d_1000_dataset_4_proj_clean_bg/landmarks/"

cases = [f"copd{i}_highres" for i in range(1, 11)]


parser = argparse.ArgumentParser()
parser.add_argument("weights_path")
parser.add_argument("--finetune", action=argparse.BooleanOptionalAction)
args = parser.parse_args()
weights_path = args.weights_path

input_shape = [1, 1, 175, 175, 175]
net = threestep_consistent_net


utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=False))
net.eval()

overall_1 = []
overall_2 = []
flips = []

for case in cases:
    image_insp = itk.imread(f"{image_root}/{case}/{case}_INSP_STD_COPD_img.nii.gz")
    image_exp = itk.imread(f"{image_root}/{case}/{case}_EXP_STD_COPD_img.nii.gz")
    seg_insp = itk.imread(f"{image_root}/{case}/{case}_INSP_STD_COPD_label.nii.gz")
    seg_exp = itk.imread(f"{image_root}/{case}/{case}_EXP_STD_COPD_label.nii.gz")

    landmarks_insp = icon_registration.test_utils.read_copd_pointset(
        landmark_root + f"/{case.split('_')[0]}_300_iBH_xyz_r1.txt"
    )
    landmarks_exp = icon_registration.test_utils.read_copd_pointset(
        landmark_root + f"/{case.split('_')[0]}_300_eBH_xyz_r1.txt"
    )

    image_insp_preprocessed = (
        icon_registration.pretrained_models.lung_network_preprocess(
            image_insp, seg_insp
        )
    )
    image_exp_preprocessed = (
        icon_registration.pretrained_models.lung_network_preprocess(image_exp, seg_exp)
    )

    phi_AB, phi_BA, loss = icon_registration.itk_wrapper.register_pair(
        net,
        image_insp_preprocessed,
        image_exp_preprocessed,
        finetune_steps=(5 if args.finetune == True else None),
        return_artifacts=True,
    )
    dists = []
    for i in range(len(landmarks_exp)):
        px, py = (
            landmarks_insp[i],
            np.array(phi_AB.TransformPoint(tuple(landmarks_exp[i]))),
        )
    dists.append(np.sqrt(np.sum((px - py) ** 2)))
    utils.log(f"Mean error on {case}: ", np.mean(dists))
    overall_1.append(np.mean(dists))
    dists = []
    for i in range(len(landmarks_insp)):
        px, py = (
            landmarks_exp[i],
            np.array(phi_BA.TransformPoint(tuple(landmarks_insp[i]))),
        )
    dists.append(np.sqrt(np.sum((px - py) ** 2)))
    utils.log(f"Mean error on {case}: ", np.mean(dists))

    overall_2.append(np.mean(dists))

    utils.log("flips:", loss.flips)

    flips.append(loss.flips)


utils.log("overall:")
utils.log(np.mean(overall_1))
utils.log(np.mean(overall_2))
utils.log("flips:", np.mean(flips))
utils.log("flips / prod(imnput_shape", np.mean(flips) / np.prod(input_shape))
