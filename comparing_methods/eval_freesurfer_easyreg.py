import subprocess
from oasis_data import get_data_list, extract_id
import nibabel as nib
import numpy as np
import os
import footsteps
footsteps.initialize(output_root="evaluation_results/")

'''
To run mri_robust_register, you need first source /playpen-raid1/tgreer/freesurfer/activate.sh
'''
def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X

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

def register_image(fixed, moving, output_folder):
    fixed_id = extract_id(fixed)
    moving_id = extract_id(moving)
    subprocess.run(f"mri_easyreg --ref {fixed} --ref_seg {output_folder}/{fixed_id}_seg.nii.gz --flo {moving} --flo_seg {output_folder}/{moving_id}_seg.nii.gz --fwd_field {output_folder}/forward_field.nii.gz --bak_field {output_folder}/backward_field.nii.gz --threads 4", shell=True)

def compute_metrics(fixed_seg, moving_seg, output_folder):
    # Compute DICE
    subprocess.run(f"mri_easywarp --i {moving_seg} --o {output_folder}/warped_moving_seg.nii.gz --field {output_folder}/forward_field.nii.gz --nearest", shell=True)

    warped_seg = load_4D(f"{output_folder}/warped_moving_seg.nii.gz")
    # warped_seg = load_4D(moving_seg)
    target_seg = load_4D(fixed_seg)
    dice_score = dice(warped_seg, target_seg)

    # Compute violation
    return dice_score

if __name__ == "__main__":
    output_folder = footsteps.output_dir
    # output_folder = "/playpen-raid2/lin.tian/projects/ByConstructionICON/evaluation_results/debug_freesurfer"

    fixed_imgs, fixed_segs, moving_imgs, moving_segs = get_data_list()

    dice_total = []
    for f, f_seg in zip(fixed_imgs, fixed_segs):
        for m, m_seg in zip(moving_imgs, moving_segs):
            f_id = extract_id(f)
            m_id = extract_id(m)
            output_folder_current = os.path.join(output_folder, f"fixed_{f_id}_moving_{m_id}")
            if not os.path.exists(output_folder_current):
                os.makedirs(output_folder_current)
            register_image(f, m, output_folder_current)
            dice_total.append(compute_metrics(f_seg, m_seg, output_folder_current))
    
    print(f"DICE: {np.array(dice_total).mean()}")