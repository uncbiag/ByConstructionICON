import argparse
import os
import random

import footsteps
footsteps.initialize(output_root="evaluation_results/")
import icon_registration as icon
import icon_registration.itk_wrapper as itk_wrapper
import itk
import numpy as np
import torch

import train_knee
import utils

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
    # image = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()(image)
    img_np = np.array(image)
    max_ = np.max(img_np)
    min_ = np.min(img_np)
    image = itk.shift_scale_image_filter(image, shift=float(min_), scale=float(1.0/(max_-min_)))
    # image = itk.clamp_image_filter(image, bounds=(0, 1))
    return image


input_shape = [1, 1, 160, 144, 192]
net = train_knee.make_net(input_shape)

#multiscale_constr_model.multiscale_affine_model
#
#qq = torch.nn.Module()
#qq.module = net
utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=True))
net.eval()

dices = []
flips = []

import glob

from comparing_methods.oasis_data import get_data_list, extract_id

fixed_imgs, fixed_segs, moving_imgs, moving_segs = get_data_list()

dice_total = []
for f, f_seg in zip(fixed_imgs, fixed_segs):
    for m, m_seg in zip(moving_imgs, moving_segs):
        image_A, image_B = (preprocess(itk.imread(n)) for n in (f, m))
        
        # import pdb; pdb.set_trace()
        phi_AB, phi_BA, loss = itk_wrapper.register_pair(
            net,
            image_A,
            image_B,
            finetune_steps=(50 if args.finetune == True else None),
            return_artifacts=True,
        )

        segmentation_A, segmentation_B = (itk.imread(n) for n in (f_seg, m_seg))

        interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_B)

        warped_segmentation_B = itk.resample_image_filter(
            segmentation_B,
            transform=phi_BA,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=segmentation_A,
        )
        mean_dice = dice(np.array(segmentation_A), np.array(warped_segmentation_B))
        # if args.writeimages:
        #     casedir = footsteps.output_dir + str(_) + "/" 
        #     os.mkdir(casedir)

        #     itk.imwrite(image_A, casedir + "imageA.nii.gz")
        #     itk.imwrite(image_B, casedir + "imageB.nii.gz")
        #     itk.imwrite(segmentation_A, casedir + "segmentation_A.nii.gz")
        #     itk.imwrite(segmentation_B, casedir + "segmentation_B.nii.gz")
        #     itk.imwrite(warped_segmentation_A, casedir+ "warpedseg.nii.gz")
        #     itk.transformwrite([phi_AB], casedir + "trans.hdf5")

        utils.log(extract_id(f), extract_id(m))
        utils.log(mean_dice)

        dice_total.append(mean_dice)
        flips.append(loss.flips)

print(f"DICE: {np.array(dice_total).mean()}")
print(f"Flips: {np.array(flips).mean()/np.prod(input_shape)*100.}")

