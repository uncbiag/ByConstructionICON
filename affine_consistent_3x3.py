import footsteps

footsteps.initialize()
import icon_registration as icon
import icon_registration.data
import icon_registration.networks as networks
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from icon_registration.config import device

import multiscale_constr_model

epochs = 10


class FlipReturn(icon_registration.RegistrationModule):
    def __init__(self, net):
        super().__init__()

        self.net = net

    def forward(self, image_A, image_B):
        return self.net(image_A, image_B), self.net(image_B, image_A)


def make_matrix_step():

    return FlipReturn(
        icon.network_wrappers.FunctionFromMatrix(
            networks.ConvolutionalMatrixNet(dimension=2)
        )
    )


def make_exp_step():

    return multiscale_constr_model.ExponentialMatrix(
        networks.ConvolutionalMatrixNet(dimension=2)
    )


def make_consistent_step():

    return multiscale_constr_model.ConsistentFromMatrix(
        networks.ConvolutionalMatrixNet(dimension=2)
    )


def one_step(step):
    return multiscale_constr_model.FirstTransform(step())


def two_step(step):
    s1 = step()
    s2 = step()

    return icon.TwoStepRegistration(
        multiscale_constr_model.FirstTransform(s1),
        multiscale_constr_model.FirstTransform(s2),
    )


def two_step_consistent(step):
    s1 = step()
    s2 = step()

    return multiscale_constr_model.FirstTransform(
        multiscale_constr_model.TwoStepInverseConsistent(s1, s2)
    )


def do_experiment():
    ds1, ds2 = icon_registration.data.get_dataset_triangles(hollow=True, data_size=30)

    curves = {}

    for inner_net in (make_matrix_step, make_exp_step, make_consistent_step):
        for step_strategy in (one_step, two_step, two_step_consistent):
            experiment_name = inner_net.__name__ + "XX" + step_strategy.__name__
            print(experiment_name)
            network = step_strategy(inner_net)

            loss = icon.losses.InverseConsistentNet(network, icon.LNCC(5), lmbda=0)
            loss.assign_identity_map((1, 1, 30, 30))

            curves[experiment_name] = multiscale_constr_model.evaluate(
                loss, experiment_name, 30, ds1=ds1
            )

    experiments = list(curves.keys())
    plt.clf()
    for i, metric_name in enumerate(curves[experiments[0]][0]._fields):

        plt.subplot(2, 3, i + 1)
        for experiment_name in experiments:
            plt.plot([getattr(val,metric_name) for val in curves[experiment_name]])

        plt.title(metric_name)
    plt.subplot(2, 3, 6)
    plt.plot([[0 for y in experiment_name]])
    plt.legend(experiments)
    plt.savefig(footsteps.output_dir + "curve_summary.pdf")
    torch.save(curves, footsteps.output_dir + "curves.trch")


do_experiment()
