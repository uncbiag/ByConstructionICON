import argparse
import os
import random

import footsteps
footsteps.initialize(output_root="evaluation_results/")
import icon_registration as icon
import icon_registration.itk_wrapper as itk_wrapper
from icon_registration.mermaidlite import compute_warped_image_multiNC, identity_map_multiN
import itk
import numpy as np
import torch
from icon_registration.losses import flips

import train_knee
import utils
import torch.nn.functional as F
import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument("weights_path" )
parser.add_argument("--finetune", action=argparse.BooleanOptionalAction)
parser.add_argument("--writeimages", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

weights_path = args.weights_path

def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    return dice/num_count

def preprocess(image):
    max_ = np.max(image)
    min_ = np.min(image)
    return (image - min_)/(max_-min_)

def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    return X

def crop_center(img, cropx, cropy, cropz):
    x, y, z = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    startz = z//2 - cropz//2
    return img[startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]

net_input_shape = [1, 1, 160, 144, 192]
device = torch.device("cuda:1")
net = train_knee.make_net(net_input_shape)

disp_scale = torch.tensor(net_input_shape[2:])[None,:,None,None,None].to(device)-1.

utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=True))
net.eval()
net.to(device)

import glob

from comparing_methods.oasis_data import get_data_list, extract_id

fixed_imgs, fixed_segs, moving_imgs, moving_segs = get_data_list()

dice_total = []
flips_total = []
violation_total = []
with torch.no_grad():
    net.eval()
    for f, f_seg in zip(fixed_imgs, fixed_segs):
        for m, m_seg in zip(moving_imgs, moving_segs):
            image_A, image_B = (preprocess(crop_center(load_4D(n), *net_input_shape[2:])) for n in (f, m))

            # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
            A_trch = torch.Tensor(image_A).to(device)[None, None]
            B_trch = torch.Tensor(image_B).to(device)[None, None]

            segmentation_A, segmentation_B = crop_center(load_4D(f_seg), *net_input_shape[2:]), torch.Tensor(crop_center(load_4D(m_seg), *net_input_shape[2:])).to(device)[None, None]
            net(A_trch, B_trch)
            phi_AB = net.phi_AB
            
            net(B_trch, A_trch)
            phi_BA = net.phi_AB
            phi_BA_vectorfield = net.phi_AB_vectorfield

            warped_seg = compute_warped_image_multiNC(segmentation_B, phi_BA_vectorfield, net.spacing, spline_order=0, zero_boundary=0).cpu().numpy()[0,0]
            
            mean_dice = dice(segmentation_A, np.array(warped_seg))
            # if args.writeimages:
            #     casedir = footsteps.output_dir + str(_) + "/" 
            #     os.mkdir(casedir)

            #     itk.imwrite(image_A, casedir + "imageA.nii.gz")
            #     itk.imwrite(image_B, casedir + "imageB.nii.gz")
            #     itk.imwrite(segmentation_A, casedir + "segmentation_A.nii.gz")
            #     itk.imwrite(segmentation_B, casedir + "segmentation_B.nii.gz")
            #     itk.imwrite(warped_segmentation_A, casedir+ "warpedseg.nii.gz")
            #     itk.transformwrite([phi_AB], casedir + "trans.hdf5")

            # violation = compute_warped_image_multiNC(phi_AB_vectorfield-net.identity_map, phi_BA_vectorfield, spacing=net.spacing, spline_order=1, zero_boundary=True) +\
            #     phi_BA_vectorfield - net.identity_map

            violation = phi_AB(phi_BA(net.identity_map)) - net.identity_map
            
            violation_total.append(
                torch.mean(torch.sqrt(torch.sum((violation*disp_scale)**2, dim=1))).item()
                )

            utils.log(extract_id(f), extract_id(m))
            utils.log(mean_dice)

            dice_total.append(mean_dice)
            flips_total.append(flips(phi_BA_vectorfield, True).item())

utils.log(f"DICE: {np.array(dice_total).mean()}")
utils.log(f"Flips(%): {np.array(flips_total).mean()}")
utils.log(f"Violations: {np.array(violation_total).mean()}")

