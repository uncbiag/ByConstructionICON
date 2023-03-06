import footsteps
footsteps.initialize()

import os

import icon_registration as icon
import icon_registration.test_utils
import icon_registration.losses as losses
import icon_registration.pretrained_models
from train_knee import make_net
import torch
import itk
import numpy as np
import icon_registration.itk_wrapper as itk_wrapper
import utils
import matplotlib.pyplot as plt

def show(im):
    plt.imshow(im, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    
def savefig(name):
    plt.savefig(name + ".png", bbox_inches="tight", pad_inches = 0)
    #plt.show()
    plt.clf()
def render_pair(name, image_A, phi_AB, image_B, net):
    
    name = footsteps.output_dir + name
    iinterpolator = itk.LinearInterpolateImageFunction.New(image_A)
    
    warped_image_A = itk.resample_image_filter(
        image_A,
        transform=phi_AB,
        interpolator=iinterpolator,
        use_reference_image=True,
        reference_image=image_B,
    )
    viewname = "Plane1"
    show(image_A[100])
    savefig(name + viewname + "image_A")
    plt.contour(net.phi_AB_vectorfield.detach().cpu()[0, 1, 50], levels=np.linspace(0, 1, 30))
    plt.contour(net.phi_AB_vectorfield.detach().cpu()[0, 2, 50], levels=np.linspace(0, 1, 30))
    show(warped_image_A[100, ::2, ::2])
    savefig(name + viewname + "warped_image_A_grid")
    
    show(warped_image_A[100])
    savefig(name + viewname + "warped_image_A")

    show(image_B[100])
    savefig(name + viewname + "image_B")

    viewname = "Plane2"
    show(image_A[:, 100])
    savefig(name + viewname + "image_A")

    plt.contour(net.phi_AB_vectorfield.detach().cpu()[0, 0, :, 50], levels=np.linspace(0, 1, 30))
    plt.contour(net.phi_AB_vectorfield.detach().cpu()[0, 2, :, 50], levels=np.linspace(0, 1, 30))
    show(warped_image_A[::2, 100, ::2])
    savefig(name + viewname + "warped_image_A_grid")

    show(warped_image_A[:, 100])
    savefig(name + viewname + "warped_image_A")

    show(image_B[:, 100])
    savefig(name + viewname + "image_B")

    
    viewname = "Plane3"
    show(image_A[:, :, 100])
    savefig(name + viewname + "image_A")

    plt.contour(net.phi_AB_vectorfield.detach().cpu()[0, 0, :, :, 50], levels=np.linspace(0, 1, 30))
    plt.contour(net.phi_AB_vectorfield.detach().cpu()[0, 1, :, :, 50], levels=np.linspace(0, 1, 30))
    show(warped_image_A[::2, ::2, 100])
    savefig(name + viewname + "warped_image_A_grid")

    show(warped_image_A[:, :, 100])
    savefig(name + viewname + "warped_image_A")

    show(image_B[:, :, 100])
    savefig(name + viewname + "image_B")


input_shape = [1, 1, 80, 192, 192]
net = make_net(input_shape)

weights_path = "results/multiscale_/network_weights_99900"

utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=False))
net.eval()

with open("../ICON/training_scripts/oai_paper_pipeline/splits/test/pair_path_list.txt") as f:
    test_pair_paths = f.readlines()[:3]



dices = []
flips = []

for i, test_pair_path in enumerate(test_pair_paths):
    test_pair_path = test_pair_path.replace("playpen", "playpen-raid").split()
    test_pair = [itk.imread(path) for path in test_pair_path]
    test_pair = [
        (
            itk.flip_image_filter(t, flip_axes=(False, False, True))
            if "RIGHT" in path
            else t
        )
        for (t, path) in zip(test_pair, test_pair_path)
    ]
    image_A, image_B, segmentation_A, segmentation_B = test_pair

    #segmentation_A = itk_half_scale_image(segmentation_A)
    #segmentation_B = itk_half_scale_image(segmentation_B)

    phi_AB, phi_BA, loss = itk_wrapper.register_pair(
        net, image_A, image_B, finetune_steps=None, return_artifacts=True
    )

    ninterpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
        segmentation_A,
        transform=phi_AB,
        interpolator=ninterpolator,
        use_reference_image=True,
        reference_image=segmentation_B,
    )
    
    
    render_pair(f"OAI{i}", image_A, phi_AB, image_B, net)
    
    mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)

    utils.log(mean_dice)
    utils.log(icon.losses.to_floats(loss))
    flips.append(loss.flips)

    dices.append(mean_dice)

utils.log("Mean DICE")
utils.log(np.mean(dices))
utils.log("Mean Flips")
utils.log(np.mean(flips))

import footsteps
import random
import argparse

import icon_registration as icon
import torch
import itk
import numpy as np
import icon_registration.itk_wrapper as itk_wrapper



def preprocess(image):
    #image = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()(image)
    max_ = np.max(np.array(image))
    image = itk.shift_scale_image_filter(image, shift=0., scale = .9 / max_)
    #image = itk.clamp_image_filter(image, bounds=(0, 1))
    return image


input_shape = [1, 1, 130, 155, 130]
net = make_net(input_shape)

weights_path = "results/brain_marc_reg-5/network_weights_99900"
utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=False))
net.eval()



dices = []

import glob
paths = glob.glob("/playpen-raid1/tgreer/Subcortical_Atlas_Fusion2/*WarpedLabels*")
atlas_registered = [p.split("/malf3")[-1].split("_")[0] for p in paths]

def get_sub_seg(n):
    path = f"/playpen-raid1/tgreer/Subcortical_Atlas_Fusion2/{n}_label.nii.gz"
    return itk.imread(path)

random.seed(1)
for i in range(3):
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
    image_A, image_B = (preprocess(itk.imread(f"/playpen-raid2/Data/HCP/HCP_1200/{n}/T1w/T1w_acpc_dc_restore_brain.nii.gz")) for n in (n_A, n_B))

    #import pdb; pdb.set_trace()
    phi_AB, phi_BA = itk_wrapper.register_pair(net, image_A, image_B, finetune_steps=None)

    segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
            segmentation_A, 
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=segmentation_B
            )
    
    render_pair(f"HCP{i}", image_A, phi_AB, image_B, net)
    
    mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)

    utils.log(mean_dice)

    dices.append(mean_dice)

utils.log("Mean DICE")
utils.log(np.mean(dices))


image_root = "/playpen-raid1/lin.tian/data/lung/dirlab_highres_350"
landmark_root = "/playpen-raid1/lin.tian/data/lung/reg_lung_2d_3d_1000_dataset_4_proj_clean_bg/landmarks/"

cases = [f"copd{i}_highres" for i in range(1, 11)]

weights_path = "results/lung_marc-2/network_weights_99900"
input_shape = [1, 1, 175, 175, 175]
net = make_net(input_shape)

utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=False))
net.eval()

overall_1 = []
overall_2 = []
flips = []

for i, case in enumerate(cases[:3]):
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
        finetune_steps=None,
        return_artifacts=True,
    )
    
    render_pair(f"Dirlab{i}", image_insp_preprocessed, phi_AB, image_exp_preprocessed, net)
    
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
