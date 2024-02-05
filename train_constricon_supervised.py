#!/usr/bin/env python

import torch
import sys
import time
import argparse
import logging
from tqdm import trange,tqdm
from utils.vxmplusplus_utils import get_vxmpp_models,return_crops
from utils.thin_plate_spline import *
from utils.data_utils import get_files

data_dir = 'data/'
dir_save = 'model/'


import network_definition

def main(args):
    
    task = args.task
    mode = 'Tr'
    do_MIND = False
    do_save = True  
    
    logging.info('Reading dataset from '+data_dir+task+'_dataset.json')
    img_fixed_all, img_moving_all, kpts_fixed_all, kpts_moving_all, case_list, orig_shapes_all, mind_fixed_all, mind_moving_all, keypts_fixed_all, img_mov_unmasked, aff_mov_all = get_files(data_dir, task, mode, do_MIND)

    logging.info('Loading model')


    fixed_img = img_fixed_all[0]
    H,W,D = fixed_img.shape[-3:]
    model = network_definition.make_net(input_shape=[1, 1, H, W, D])
    print(H, W, D)
    
    model.train()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

    for repeat in range(2):
        if repeat == 0:
            num_iterations = 1*4900
        else:
            num_iterations = 7*4900
        t0 = time.time()
        run_tre = torch.empty(0,1); run_tre_test = torch.empty(0,1); 
        run_loss = torch.zeros(num_iterations)

        with tqdm(total=num_iterations, file=sys.stdout) as pbar:

            for i in range(num_iterations):
                fixed_imgs = []
                moving_imgs =[]

                for _ in range(2): # BATCH SIZE
                    ii = torch.randperm(len(img_fixed_all))[0]
                    
                    fixed_img = img_fixed_all[ii]
                    moving_img = img_moving_all[ii]

                    fixed_img = fixed_img.view(1,1,H,W,D).cuda()
                    moving_img = moving_img.view(1,1,H,W,D).cuda()

                    fixed_imgs.append(fixed_img)
                    moving_imgs.append(moving_img)

                fixed_img = torch.cat(fixed_imgs, dim=0)
                moving_img = torch.cat(moving_imgs, dim=0)


                fixed_img = fixed_img - torch.min(fixed_img)
                moving_img = moving_img - torch.min(moving_img)

                moving_img = moving_img.float() / torch.max(moving_img)
                fixed_img = fixed_img.float() / torch.max(fixed_img)

                moving_img, fixed_img = network_definition.augment(moving_img, fixed_img)

                optimizer.zero_grad()

                loss_object = model(fixed_img, moving_img)

                loss_object.all_loss.backward()
                optimizer.step()


                run_loss[i] = loss_object.all_loss.item()

                str1 = f"iter: {i}, last_loss: {'%0.3f'%loss_object.all_loss.item()}, loss: {'%0.3f'%(run_loss[i-28:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t0)} sec, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
                logging.info(str1)
               
        if(repeat==0):
            torch.save(model.state_dict(), dir_save + 'constricon_0.pth')
        else:
            logging.info('Saving model')
            torch.save(model.state_dict(), dir_save + 'constricon.pth')        


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",\
        handlers=[logging.FileHandler(dir_save+"debug.log"),logging.StreamHandler()])

    parser = argparse.ArgumentParser(description = 'Training of VoxelMorph++')
    parser.add_argument('task', default='ThoraxCBCT', help="task/dataset: ThoraxCBCT or OncoReg")
    args = parser.parse_args()
    main(args)






