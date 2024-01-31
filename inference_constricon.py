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

data_dir = 'data/'
model = 'model/vxmpp.pth'
outfolder = 'results/'

def main(args):
    
    task = args.task
    mode = args.mode

    do_MIND = True
    
    logging.info('Loading data')
    img_fixed_all, img_moving_all, kpts_fixed_all, kpts_moving_all, case_list, orig_shapes_all, mind_fixed_all, mind_moving_all, keypts_fixed_all, img_mov_unmasked, aff_mov_all = get_files(data_dir, task, mode, do_MIND)

    unet_model,heatmap,mesh = get_vxmpp_models()

    logging.info('Loading model')
    state_dicts = torch.load(model)
    unet_model.load_state_dict(state_dicts[1])
    heatmap.load_state_dict(state_dicts[0])

    predictions = []
    
    for case in trange(len(case_list)):
        ##MASKED INPUT IMAGES ARE HALF-RESOLUTION
        with torch.no_grad():
            fixed_img = img_fixed_all[case]
            moving_img = img_moving_all[case]
            keypts_fix = keypts_fixed_all[case].squeeze().cuda()
            H,W,D = fixed_img.shape[-3:]

            fixed_img = fixed_img.view(1,1,H,W,D).cuda()
            moving_img = moving_img.view(1,1,H,W,D).cuda()

            with torch.cuda.amp.autocast():
                #VoxelMorph requires some padding
                input,x_start,y_start,z_start,x_end,y_end,z_end = return_crops(torch.cat((fixed_img,moving_img),1).cuda())
                output = F.pad(F.interpolate(unet_model(input),scale_factor=2),(z_start,(-z_end+D),y_start,(-y_end+W),x_start,(-x_end+H)))
                disp_est = torch.zeros_like(keypts_fix)
                for idx in torch.split(torch.arange(len(keypts_fix)),1024):
                    sample_xyz = keypts_fix[idx]
                    sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear')
                    disp_pred = heatmap(sampled.permute(2,1,0,3,4))
                    disp_est[idx] = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)


        ##NOW EVERYTHING FULL-RESOLUTION
        H,W,D = orig_shapes_all[case]

        fixed_mind = mind_fixed_all[case].view(1,-1,H//2,W//2,D//2).cuda()
        moving_mind = mind_moving_all[case].view(1,-1,H//2,W//2,D//2).cuda()

        pred_xyz,disp_smooth,dense_flow = adam_mind(keypts_fix,disp_est,fixed_mind,moving_mind,H,W,D)
        predictions.append(pred_xyz.cpu()+keypts_fix.cpu())


    torch.save({'keypts_mov_predict':predictions,'case_list':case_list,'keypts_fix':keypts_fixed_all},outfolder + '/predictions.pth')
    if(outfolder is not None):
        for i in range(len(case_list)):
            logging.info('Case:'+str(i))
            case = case_list[i]
            output_path = outfolder+'/'+case
            H,W,D = orig_shapes_all[i]
            kpts_fix = torch.flip(keypts_fixed_all[i].squeeze(),(1,))*torch.tensor([H/2,W/2,D/2])+torch.tensor([H/2,W/2,D/2])
            kpts_moved = torch.flip(predictions[i].squeeze(),(1,))*torch.tensor([H/2,W/2,D/2])+torch.tensor([H/2,W/2,D/2])
            np.savetxt('{}.csv'.format(output_path), torch.cat([kpts_fix, kpts_moved], dim=1).cpu().numpy(), delimiter=",", fmt='%.3f')
            logging.info('Keypoints saved')

            img_mov = img_mov_unmasked[i]
            aff_mov = aff_mov_all[i]

            cf = torch.from_numpy(np.loadtxt(outfolder+'/'+case+'.csv',delimiter=',')).float()
            kpts_fixed = torch.flip((cf[:,:3]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))
            kpts_moving = torch.flip((cf[:,3:]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))
            with torch.no_grad():
                dense_flow = thin_plate_dense(kpts_fixed.unsqueeze(0).cuda(), (kpts_moving-kpts_fixed).unsqueeze(0).cuda(), (H, W, D), 4, 0.001)
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






