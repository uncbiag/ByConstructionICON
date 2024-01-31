#!/usr/bin/env python

import torch
import torch.nn.functional as F
from tqdm import trange
import argparse
import numpy as np
import nibabel as nib
import logging 
from scipy.ndimage.interpolation import zoom as zoom
from utils.vxmplusplus_utils import adam_mind,get_vxmpp_models,return_crops
from utils.thin_plate_spline import thin_plate_dense
from utils.data_utils import get_files


import network_definition

data_dir = 'data/'
model_weights_path = 'model/constricon.pth'
outfolder = 'results/'

def main(args):
    
    task = args.task
    mode = args.mode

    do_MIND = False
    
    logging.info('Loading data')
    img_fixed_all, img_moving_all, kpts_fixed_all, kpts_moving_all, case_list, orig_shapes_all, mind_fixed_all, mind_moving_all, keypts_fixed_all, img_mov_unmasked, aff_mov_all = get_files(data_dir, task, mode, do_MIND)

    fixed_img = img_fixed_all[0]
    H,W,D = fixed_img.shape[-3:]
    print(H, W, D)
    model = network_definition.make_net(input_shape=[1, 1, H, W, D])

    logging.info('Loading model')
    state_dict = torch.load(model_weights_path)
    model.load_state_dict(state_dict)

    dense_flows = []
    
    for case in trange(len(case_list)):
        with torch.no_grad():
            fixed_img = img_fixed_all[case]
            moving_img = img_moving_all[case]
            H,W,D = fixed_img.shape[-3:]

            fixed_img = fixed_img.view(1,1,H,W,D).cuda()
            moving_img = moving_img.view(1,1,H,W,D).cuda()


        H,W,D = orig_shapes_all[case]
        print(H, W, D)

        state_dict = torch.load(model_weights_path)
        model.load_state_dict(state_dict)

        fixed_img = fixed_img - torch.min(fixed_img)
        moving_img = moving_img - torch.min(moving_img)

        moving_img = moving_img.float() / torch.max(moving_img)
        fixed_img = fixed_img.float() / torch.max(fixed_img)
        model.train()
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
        for _ in range(50):
            optimizer.zero_grad()
            loss_tuple = model(moving_img, fixed_img )
            loss_tuple.all_loss.backward()
            optimizer.step()
            print(loss_tuple.all_loss.item())
        dense_flow = (model.phi_AB_vectorfield - model.identity_map)
        dense_flow = F.interpolate(dense_flow,scale_factor=2,mode='trilinear')
        dense_flow = dense_flow.permute(0, 2, 3, 4, 1)[:, :, :, :, [2, 1, 0]].detach().cpu() * 2.
        dense_flows.append(dense_flow)


    if(outfolder is not None):
        for i in range(len(case_list)):
            logging.info('Case:'+str(i))
            case = case_list[i]
            output_path = outfolder+'/'+case
            H,W,D = orig_shapes_all[i]

            img_mov = img_mov_unmasked[i]
            aff_mov = aff_mov_all[i]

            dense_flow = dense_flows[i]
            print(img_mov.shape, dense_flow.shape, H, W, D)

            warped_img = F.grid_sample(img_mov.view(1,1,H,W,D),dense_flow.cpu()+F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D))).squeeze()
            warped = nib.Nifti1Image(warped_img.numpy(), aff_mov)  
            nib.save(warped, outfolder + '/warped_' + case.split('T_')[1] + '.nii.gz')
            logging.info('Warped image saved')
            
            dense_flow = dense_flow.cpu().flip(4).permute(0, 4, 1, 2, 3) * torch.tensor( [H - 1, W - 1, D - 1]).view(1, 3, 1, 1, 1) / 2
            grid_sp = 1
            disp_lr = F.interpolate(dense_flow, size=(H // grid_sp, W // grid_sp, D // grid_sp), mode='trilinear',
                                                align_corners=False)
            disp_lr = disp_lr.permute(0,2,3,4,1)
            disp_tmp = disp_lr[0].permute(3,0,1,2).numpy()
            disp_lr = disp_lr[0].numpy()
            displacement_field = nib.Nifti1Image(disp_lr, aff_mov)
            nib.save(displacement_field, outfolder + '/disp_' + case.split('T_')[1] + '_' +case.split('_')[1] + '_0000.nii.gz')
            logging.info('Displacement field saved')
             

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",\
        handlers=[logging.FileHandler(outfolder+"debug.log"),logging.StreamHandler()])
    
    parser = argparse.ArgumentParser(description = 'Inference of VoxelMorph++')
    parser.add_argument('task',      default='ThoraxCBCT', help="task/dataset: ThoraxCBCT or OncoReg")
    parser.add_argument('mode',      default='Ts', help="Run inference on validation ('Val') or test ('Ts') data")
    args = parser.parse_args()
    main(args)






