
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
import json
from utils.foerstner import foerstner_kpts 
from utils.vxmplusplus_utils import MINDSSC


def get_files(data_dir, task, mode, do_MIND):

    if task == "ThoraxCBCT" or task == "OncoReg":
        data_json = os.path.join(data_dir, task + "_dataset.json")
        with open(data_json) as file:
            data = json.load(file)

        if mode == 'Tr':
            mode1 = 'training_paired_images'
        elif mode == 'Val':
            mode = 'Tr'
            mode1 = 'registration_val'
        elif mode == 'Ts':
            mode1 = 'registration_test'

        img_fixed_all = []
        img_moving_all = []
        kpts_fixed_all = []
        kpts_moving_all = []
        orig_shapes_all = []
        mind_fixed_all = []
        mind_moving_all = []
        case_list = []
        keypts_fixed_all = []
        img_mov_unmasked = []
        aff_mov_all = []

        for pair in data[mode1]:
            nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
            nam_moving = os.path.basename(pair["moving"]).split(".")[0]
            if nam_fixed.split('.nii.gz')[0].split('_')[2]=='0001':
                kpts_dir = data_dir + 'keypoints01'
            else:
                kpts_dir = data_dir + 'keypoints02'
     
            case_list.append(nam_fixed)

            img_fixed = torch.from_numpy(nib.load(os.path.join(data_dir, "images" + mode, nam_fixed + ".nii.gz")).get_fdata()).float()
            img_moving = torch.from_numpy(nib.load(os.path.join(data_dir, "images" + mode, nam_moving + ".nii.gz")).get_fdata()).float()
            aff_mov = nib.load(os.path.join(data_dir, "images" + mode, nam_moving + ".nii.gz")).affine
            #"fixed_label": os.path.join('/home/heyer/storage/staff/wiebkeheyer/data/ThoraxCBCT/additional_data/TSv2/ml_13' , nam_fixed + ".nii.gz"),
            #"moving_label": os.path.join('/home/heyer/storage/staff/wiebkeheyer/data/ThoraxCBCT/additional_data/TSv2/ml_13' , nam_moving + ".nii.gz"),
            label_fixed = torch.from_numpy(nib.load(os.path.join(data_dir, 'masks' + mode, nam_fixed + ".nii.gz")).get_fdata()).float()
            label_moving = torch.from_numpy(nib.load(os.path.join(data_dir, 'masks' + mode, nam_moving + ".nii.gz")).get_fdata()).float()
            kpts_fixed = torch.from_numpy(np.loadtxt(os.path.join(kpts_dir + mode, nam_fixed + ".csv"),delimiter=',')).float()
            kpts_moving = torch.from_numpy(np.loadtxt(os.path.join(kpts_dir + mode, nam_moving + ".csv"),delimiter=',')).float()

            masked_fixed = F.interpolate(((img_fixed+1024)*label_fixed).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()
            masked_moving = F.interpolate(((img_moving+1024)*label_moving).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()

            shape = label_fixed.shape

            kpts_fix = foerstner_kpts(img_fixed.unsqueeze(0).unsqueeze(0).cuda(), label_fixed.unsqueeze(0).unsqueeze(0).cuda(), 1.4, 3).cpu()
            keypts_fixed_all.append(kpts_fix)

            img_fixed_all.append(masked_fixed)
            img_moving_all.append(masked_moving)
            kpts_fixed_all.append(kpts_fixed)
            kpts_moving_all.append(kpts_moving)
            orig_shapes_all.append(shape)
            img_mov_unmasked.append(img_moving)
            aff_mov_all.append(aff_mov)

            if(do_MIND):
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        mind_fixed = F.avg_pool3d(label_fixed.unsqueeze(0).unsqueeze(0).cuda().half()*\
                            MINDSSC(img_fixed.unsqueeze(0).unsqueeze(0).cuda(),1,2).half(),2).cpu()
                        mind_moving = F.avg_pool3d(label_moving.unsqueeze(0).unsqueeze(0).cuda().half()*\
                            MINDSSC(img_moving.unsqueeze(0).unsqueeze(0).cuda(),1,2).half(),2).cpu()

                mind_fixed_all.append(mind_fixed)
                mind_moving_all.append(mind_moving)
                del mind_fixed
                del mind_moving
                     
    else:
        raise ValueError(f"Task {task} undefined!")
    
    return img_fixed_all, img_moving_all, kpts_fixed_all, kpts_moving_all, case_list, orig_shapes_all, mind_fixed_all, mind_moving_all, keypts_fixed_all, img_mov_unmasked, aff_mov_all
